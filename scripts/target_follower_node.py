"""
target_follower_node.py — 파트2·3 통합의 핵심 신규 노드 (Week 4)
================================================================
obstacle_locator_node 가 발행하는 "로봇 기준 거리·방위각"을 받아
TurtleBot3 를 목표 지점까지 이동시킨다.

제어 로직 (멘토링 회의에서 정한 단순 규칙):
  - |방위각| > 5°          → 제자리 회전 (방위각 부호 방향으로)
  - |방위각| ≤ 5°          → 전진 (+ 미세 방향 보정)
  - 거리 ≤ 0.35 m          → 정지 (도착) ← Week3 특이점(거리≈0에서 방위각
                             불안정) 대응: 도착 판정 후엔 방위각 무시
  - 데이터 끊김(2초) 시    → 안전 정지

입력:  /obstacle/range_bearing_robot (geometry_msgs/Point, x=거리[m], y=방위각[deg])
출력:  /cmd_vel (geometry_msgs/Twist)

부호 규약: 방위각 + = 목표가 로봇 왼쪽 → angular.z + (좌회전). REP-103 일치.

사용 예:
  python3 target_follower_node.py
  python3 target_follower_node.py --ros-args -p max_lin:=0.1 -p arrive_dist:=0.4

담당: 정우열 · 프로젝트: 스마트해운물류 x ICT 멘토링 (갯끈풀 탐지, 2번 파트)
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point, Twist


class TargetFollowerNode(Node):
    def __init__(self):
        super().__init__("target_follower_node")

        # ── 파라미터 ──
        self.declare_parameter("bearing_tol_deg", 5.0)   # 이 안이면 "정면" 판정
        self.declare_parameter("arrive_dist", 0.35)      # 도착 판정 거리 [m]
        self.declare_parameter("max_lin", 0.15)          # 최대 전진 속도 [m/s] (burger 한계 0.22)
        self.declare_parameter("max_ang", 0.5)           # 최대 회전 속도 [rad/s]
        self.declare_parameter("k_lin", 0.3)             # 전진 P게인 (속도 = k_lin × 거리)
        self.declare_parameter("k_ang", 0.03)            # 회전 P게인 (rad/s per deg)
        self.declare_parameter("data_timeout", 2.0)      # 입력 끊김 판정 [s]

        self.bearing_tol = float(self.get_parameter("bearing_tol_deg").value)
        self.arrive_dist = float(self.get_parameter("arrive_dist").value)
        self.max_lin = float(self.get_parameter("max_lin").value)
        self.max_ang = float(self.get_parameter("max_ang").value)
        self.k_lin = float(self.get_parameter("k_lin").value)
        self.k_ang = float(self.get_parameter("k_ang").value)
        self.data_timeout = float(self.get_parameter("data_timeout").value)

        # ── 상태 ──
        self.distance = None       # 최근 수신한 로봇 기준 거리 [m]
        self.bearing_deg = None    # 최근 수신한 로봇 기준 방위각 [deg]
        self.last_msg_time = None  # 마지막 수신 시각
        self.arrived = False       # 도착 래치 (덜덜거림 방지)

        self.create_subscription(
            Point, "/obstacle/range_bearing_robot", self.on_range_bearing, 10
        )
        self.pub_cmd = self.create_publisher(Twist, "/cmd_vel", 10)

        self.timer = self.create_timer(0.1, self.on_timer)  # 10 Hz 제어 루프
        self.get_logger().info(
            f"target_follower 시작 (허용각 ±{self.bearing_tol}°, "
            f"도착거리 {self.arrive_dist}m, 최대속도 {self.max_lin}m/s)"
        )

    # ─────────────────────────────────────────
    def on_range_bearing(self, msg: Point):
        self.distance = float(msg.x)
        self.bearing_deg = float(msg.y)
        self.last_msg_time = self.get_clock().now()

    def _stop(self):
        self.pub_cmd.publish(Twist())  # 모든 필드 0

    @staticmethod
    def _clamp(v, limit):
        return max(-limit, min(limit, v))

    # ─────────────────────────────────────────
    def on_timer(self):
        # 1) 입력이 아직 없거나 끊겼으면 정지 (안전)
        if self.last_msg_time is None:
            self._stop()
            return
        age = (self.get_clock().now() - self.last_msg_time).nanoseconds * 1e-9
        if age > self.data_timeout:
            self.get_logger().warn(
                f"입력 끊김 ({age:.1f}s) → 정지", throttle_duration_sec=5.0
            )
            self._stop()
            return

        dist, bearing = self.distance, self.bearing_deg

        # 2) 도착 판정 (래치: 도착 후 거리가 1.5배 이상 벌어져야 재출발)
        if self.arrived:
            if dist > self.arrive_dist * 1.5:
                self.arrived = False
                self.get_logger().info("목표가 다시 멀어짐 → 재출발")
            else:
                self._stop()
                return
        elif dist <= self.arrive_dist:
            self.arrived = True
            self.get_logger().info(f"★ 도착! (거리 {dist:.2f}m) → 정지")
            self._stop()
            return

        # 3) 회전/전진 제어
        cmd = Twist()
        if abs(bearing) > self.bearing_tol:
            # 방위각이 크면 제자리 회전만
            cmd.angular.z = self._clamp(self.k_ang * bearing, self.max_ang)
        else:
            # 정면이면 전진 + 미세 보정
            cmd.linear.x = self._clamp(self.k_lin * dist, self.max_lin)
            cmd.angular.z = self._clamp(self.k_ang * bearing, self.max_ang)

        self.pub_cmd.publish(cmd)
        self.get_logger().info(
            f"거리={dist:.2f}m 방위={bearing:+.1f}° → "
            f"lin={cmd.linear.x:.2f} ang={cmd.angular.z:+.2f}",
            throttle_duration_sec=1.0,
        )


def main():
    rclpy.init()
    node = TargetFollowerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node._stop()  # 종료 시 로봇 정지
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
