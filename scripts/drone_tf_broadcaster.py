"""
drone_tf_broadcaster.py — [Phase 1] 정적 TF → 동적 TF 전환
============================================================
기존 문제:
  spartina_nav2.launch.py 가 map→drone_camera_optical 을
  static_transform_publisher 로 발행 → 드론을 아무리 움직여도
  TF 는 (0,0,8) 이라고 계속 우김 → 투영 좌표가 틀어짐.

이 노드가 하는 일:
  Gazebo 의 drone_camera 모델 실제 pose 를 /gazebo/model_states 로 읽어서
  map→drone_camera_optical TF 를 실시간으로 발행한다.

  기존 static_transform_publisher 를 이 노드로 "그냥 갈아끼우는" 것이므로
  obstacle_locator_node / goal_relay / Nav2 는 코드를 한 줄도 안 고쳐도 된다.
  (전부 이미 TF2 로 조회하도록 짜여 있기 때문)

좌표계 설명 (여기가 유일하게 헷갈리는 부분):
  ① Gazebo 모델 프레임 : world 파일의 <pose>0 0 8 0 1.5708 0</pose>
                          = pitch 90° 회전. 이 프레임에서 카메라는 +X 를 본다.
                          (Gazebo 카메라 규약: x=앞, y=왼쪽, z=위)
  ② ROS 광학 프레임    : z=앞, x=오른쪽, y=아래  ← pixel_to_ground 가 쓰는 규약
  ①→② 는 항상 똑같은 고정 회전이고, 그 쿼터니언이 아래 Q_CAM2OPT 다.

  검증 완료: 모델 pose(pitch 90°) ⊗ Q_CAM2OPT 를 계산하면
             R = [[0,-1,0],[-1,0,0],[0,0,-1]] 이 나오는데,
             이건 기존 런치의 static TF 및 obstacle_locator 의
             R_HARDCODED 와 정확히 일치한다.
             → 드론이 제자리에 있을 때는 기존과 100% 동일하게 동작하고,
               움직이면 그때부터 따라간다.

전제:
  map = odom = Gazebo 원점 (기존 런치가 identity 로 발행) 이므로
  Gazebo 모델 pose 를 그대로 map 기준 pose 로 쓸 수 있다.

실행:
  python3 drone_tf_broadcaster.py --ros-args -p use_sim_time:=true

담당: 정우열 · 스마트해운물류 x ICT 멘토링 (갯끈풀 탐지, 2번 파트)
"""

import rclpy
from rclpy.node import Node
from gazebo_msgs.msg import ModelStates
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster


# Gazebo 카메라 프레임 → ROS 광학 프레임 고정 변환 (x,y,z,w)
# 표준값이며, 이 프로젝트에서 수치로 검증됨 (위 주석 참고)
Q_CAM2OPT = (-0.5, 0.5, -0.5, 0.5)


def quat_mul(q1, q2):
    """쿼터니언 곱 q1 ⊗ q2. 입력·출력 모두 (x, y, z, w) 순서."""
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return (
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
    )


class DroneTfBroadcaster(Node):
    def __init__(self):
        super().__init__("drone_tf_broadcaster")

        self.declare_parameter("model_name", "drone_camera")
        self.declare_parameter("parent_frame", "map")
        self.declare_parameter("child_frame", "drone_camera_optical")
        # 모델 프레임이 이미 카메라를 향하고 있으면 true (지금 world 가 이 경우).
        # 순수 위치만 있는 드론 본체 모델을 쓰게 되면 false 로 두고 직접 자세를 줘야 한다.
        self.declare_parameter("apply_optical_rotation", True)

        self.model_name = self.get_parameter("model_name").value
        self.parent_frame = self.get_parameter("parent_frame").value
        self.child_frame = self.get_parameter("child_frame").value
        self.apply_optical = bool(self.get_parameter("apply_optical_rotation").value)

        self.br = TransformBroadcaster(self)
        self.create_subscription(ModelStates, "/gazebo/model_states", self.on_states, 10)

        self._warned_missing = False
        self._count = 0

        self.get_logger().info(
            f"drone_tf_broadcaster 시작 — 모델 '{self.model_name}' 의 pose 를 "
            f"{self.parent_frame} → {self.child_frame} TF 로 발행합니다.\n"
            f"  ※ 런치에서 static_transform_publisher(map→{self.child_frame}) 를 "
            f"반드시 꺼야 합니다. 둘 다 켜면 TF 가 충돌합니다."
        )

    def on_states(self, msg: ModelStates):
        if self.model_name not in msg.name:
            if not self._warned_missing:
                self.get_logger().error(
                    f"Gazebo 에 '{self.model_name}' 모델이 없습니다. "
                    f"현재 모델 목록: {list(msg.name)}"
                )
                self._warned_missing = True
            return
        self._warned_missing = False

        pose = msg.pose[msg.name.index(self.model_name)]

        q_model = (pose.orientation.x, pose.orientation.y,
                   pose.orientation.z, pose.orientation.w)
        q_out = quat_mul(q_model, Q_CAM2OPT) if self.apply_optical else q_model

        t = TransformStamped()
        # ModelStates 에는 header 가 없어서 현재 시각으로 찍는다.
        # (use_sim_time:=true 로 켜야 Gazebo 시계를 따라간다)
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = self.parent_frame
        t.child_frame_id = self.child_frame
        t.transform.translation.x = pose.position.x
        t.transform.translation.y = pose.position.y
        t.transform.translation.z = pose.position.z
        t.transform.rotation.x = q_out[0]
        t.transform.rotation.y = q_out[1]
        t.transform.rotation.z = q_out[2]
        t.transform.rotation.w = q_out[3]
        self.br.sendTransform(t)

        # 5초에 한 번만 현재 위치 로그 (터미널 도배 방지)
        self._count += 1
        if self._count % 250 == 1:
            self.get_logger().info(
                f"드론 pose = ({pose.position.x:+.2f}, "
                f"{pose.position.y:+.2f}, {pose.position.z:+.2f})"
            )


def main():
    rclpy.init()
    node = DroneTfBroadcaster()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
