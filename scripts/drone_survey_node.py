"""
drone_survey_node.py — [Phase 2] 드론 자율 탐사 비행 (지그재그 순회)
====================================================================
드론을 /gazebo/set_entity_state 서비스로 웨이포인트를 따라 옮긴다.

왜 비행 물리를 안 쓰나:
  쿼드로터 동역학(PX4 SITL 등)은 그 자체로 별도 프로젝트다.
  우리 프로젝트에서 드론은 "정해진 고도에서 이동하는 카메라"일 뿐이고,
  검증 대상은 '탐지 → 좌표 변환 → 로봇 주행' 파이프라인이다.
  따라서 위치를 부드럽게 보간해 옮기는 것으로 충분하고,
  이렇게 하면 이륙/착륙/자세제어 튜닝에 시간을 안 뺏긴다.
  (보고서에는 "비행 제어는 범위 밖, 위치는 이상적으로 주어진다고 가정"으로 명시)

동작:
  1. lawnmower(지그재그) 경로를 자동 생성
  2. 각 웨이포인트 사이를 speed[m/s] 로 선형 보간해 이동
  3. 웨이포인트 도착 시 hold_sec 만큼 정지 (영상이 흔들리지 않은 상태에서 탐지)
  4. 정지 중일 때 /drone/survey_state 에 "HOLD", 이동 중엔 "MOVING" 발행

  → Phase 4 탐지기는 "HOLD 일 때만 탐지"하게 만들면
    모션블러 + 동기화 오차를 동시에 피할 수 있다.

파라미터:
  area_x_min/max, area_y_min/max : 탐사 영역 [m]
  lane_spacing : 지그재그 간격 [m] — 카메라 지상 시야폭보다 좁게 잡아야 빈틈이 없다
                 고도 8m, HFOV 60° 기준 지상 가로 시야 ≈ 2·8·tan(30°) ≈ 9.24 m
                 → 30% 겹침을 주려면 lane_spacing ≈ 6.5 m
  altitude : 비행 고도 [m]
  speed    : 이동 속도 [m/s]
  hold_sec : 웨이포인트 정지 시간 [s]
  loop     : 경로 끝에서 처음으로 돌아갈지

실행:
  python3 drone_survey_node.py --ros-args -p use_sim_time:=true \
    -p area_x_min:=-4.0 -p area_x_max:=4.0 \
    -p area_y_min:=-3.0 -p area_y_max:=3.0 -p lane_spacing:=3.0

담당: 정우열 · 스마트해운물류 x ICT 멘토링 (갯끈풀 탐지, 2번 파트)
"""

import math

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from gazebo_msgs.srv import SetEntityState
from gazebo_msgs.msg import EntityState


# world 파일의 drone_camera <pose> 마지막 3개 = rpy(0, 1.5708, 0) 에 해당하는 쿼터니언.
# 카메라를 계속 수직 하방으로 유지하기 위해 자세는 고정한다.
Q_NADIR = (0.0, 0.70710678, 0.0, 0.70710678)  # (x, y, z, w)


def make_lawnmower(x_min, x_max, y_min, y_max, spacing):
    """지그재그(lawnmower) 경로 생성. 왕복하며 y 를 spacing 만큼씩 이동."""
    pts = []
    y = y_min
    left_to_right = True
    while y <= y_max + 1e-6:
        if left_to_right:
            pts.append((x_min, y))
            pts.append((x_max, y))
        else:
            pts.append((x_max, y))
            pts.append((x_min, y))
        left_to_right = not left_to_right
        y += spacing
    return pts


class DroneSurveyNode(Node):
    def __init__(self):
        super().__init__("drone_survey_node")

        self.declare_parameter("model_name", "drone_camera")
        self.declare_parameter("area_x_min", -4.0)
        self.declare_parameter("area_x_max", 4.0)
        self.declare_parameter("area_y_min", -3.0)
        self.declare_parameter("area_y_max", 3.0)
        self.declare_parameter("lane_spacing", 3.0)
        self.declare_parameter("altitude", 8.0)
        self.declare_parameter("speed", 0.8)
        self.declare_parameter("hold_sec", 3.0)
        self.declare_parameter("rate_hz", 20.0)
        self.declare_parameter("loop", True)

        g = lambda n: self.get_parameter(n).value
        self.model_name = g("model_name")
        self.altitude = float(g("altitude"))
        self.speed = float(g("speed"))
        self.hold_sec = float(g("hold_sec"))
        self.rate_hz = float(g("rate_hz"))
        self.loop = bool(g("loop"))

        self.waypoints = make_lawnmower(
            float(g("area_x_min")), float(g("area_x_max")),
            float(g("area_y_min")), float(g("area_y_max")),
            float(g("lane_spacing")),
        )

        self.cli = self.create_client(SetEntityState, "/gazebo/set_entity_state")
        self.pub_state = self.create_publisher(String, "/drone/survey_state", 10)

        # 진행 상태
        self.idx = 0                 # 향하고 있는 웨이포인트 인덱스
        self.pos = list(self.waypoints[0])   # 현재 위치 (x, y)
        self.hold_left = self.hold_sec
        self.phase = "HOLD"          # HOLD | MOVING | DONE

        self.get_logger().info(
            f"drone_survey_node 시작 — 웨이포인트 {len(self.waypoints)}개, "
            f"고도 {self.altitude} m, 속도 {self.speed} m/s\n"
            f"  경로: {self.waypoints}"
        )

        if not self.cli.wait_for_service(timeout_sec=10.0):
            self.get_logger().error(
                "/gazebo/set_entity_state 서비스를 찾을 수 없습니다.\n"
                "  → world 파일에 libgazebo_ros_state.so 플러그인이 들어있는지 확인하세요.\n"
                "  → spartina_world_moving.world 를 쓰고 있는지 확인하세요."
            )

        self.timer = self.create_timer(1.0 / self.rate_hz, self.on_timer)

    def teleport(self, x, y):
        req = SetEntityState.Request()
        st = EntityState()
        st.name = self.model_name
        st.pose.position.x = float(x)
        st.pose.position.y = float(y)
        st.pose.position.z = float(self.altitude)
        st.pose.orientation.x = Q_NADIR[0]
        st.pose.orientation.y = Q_NADIR[1]
        st.pose.orientation.z = Q_NADIR[2]
        st.pose.orientation.w = Q_NADIR[3]
        st.reference_frame = "world"
        req.state = st
        self.cli.call_async(req)   # 응답을 기다리지 않는다 (20Hz 유지)

    def on_timer(self):
        dt = 1.0 / self.rate_hz

        if self.phase == "DONE":
            self.pub_state.publish(String(data="DONE"))
            return

        if self.phase == "HOLD":
            self.hold_left -= dt
            self.pub_state.publish(String(data="HOLD"))
            if self.hold_left <= 0.0:
                # 다음 웨이포인트로 출발
                self.idx += 1
                if self.idx >= len(self.waypoints):
                    if self.loop:
                        self.idx = 0
                    else:
                        self.phase = "DONE"
                        self.get_logger().info("탐사 경로 완료")
                        return
                self.phase = "MOVING"
                self.get_logger().info(
                    f"→ 웨이포인트 {self.idx}/{len(self.waypoints)-1} "
                    f"{self.waypoints[self.idx]} 로 이동"
                )
            self.teleport(*self.pos)
            return

        # MOVING
        tx, ty = self.waypoints[self.idx]
        dx, dy = tx - self.pos[0], ty - self.pos[1]
        dist = math.hypot(dx, dy)
        step = self.speed * dt

        if dist <= step or dist < 1e-6:
            self.pos = [tx, ty]
            self.phase = "HOLD"
            self.hold_left = self.hold_sec
            self.get_logger().info(f"  도착 ({tx:+.2f}, {ty:+.2f}) — {self.hold_sec}s 정지")
        else:
            self.pos[0] += dx / dist * step
            self.pos[1] += dy / dist * step

        self.pub_state.publish(String(data="MOVING"))
        self.teleport(*self.pos)


def main():
    rclpy.init()
    node = DroneSurveyNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
