"""
goal_relay_node.py — 파트2·3 연동 다리 (변환 노드)
====================================================
파트2(obstacle_locator_node)가 발행하는 "목표의 world 절대좌표"를
파트3 Nav2가 이해하는 "목표 자세(/goal_pose)"로 바꿔주는 노드.

  파트2 /obstacle/ground_point (geometry_msgs/Point, world 기준 x,y)
       │  ↓  [이 노드]
  파트3 /goal_pose (geometry_msgs/PoseStamped, map 프레임)
       │  ↓
  Nav2 → 장애물 피해 목표까지 자율주행

핵심 설계 — 왜 "변화가 있을 때만" 발행하나:
  obstacle_locator 는 목표 좌표를 2Hz로 "계속" 뱉는다. 그걸 그대로
  /goal_pose 로 흘리면 Nav2 가 0.5초마다 목표를 새로 받아 경로를 계속
  리셋해서 로봇이 제자리에서 버벅인다. 그래서 목표가 일정 거리(기본 0.15m)
  이상 "새로 바뀌었을 때만" 한 번 발행한다. (클릭으로 목표를 바꾸면 좌표가
  점프 → 새 목표 발행)

파라미터:
  - input_topic   (기본 /obstacle/ground_point) : 파트2가 주는 world 좌표
  - goal_frame    (기본 map)                     : Nav2 global_frame
  - move_threshold(기본 0.15) [m]               : 이만큼 바뀌어야 새 목표로 인정
  - face_goal     (기본 True)                    : True면 목표를 향하도록 yaw 설정

사용 예:
  python3 goal_relay_node.py
  python3 goal_relay_node.py --ros-args -p goal_frame:=map -p move_threshold:=0.2

담당: 파트2·3 통합 · 프로젝트: 스마트해운물류 x ICT 멘토링 (갯끈풀 탐지)
"""

import math

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy
from geometry_msgs.msg import Point, PoseStamped


class GoalRelayNode(Node):
    def __init__(self):
        super().__init__("goal_relay_node")

        # ── 파라미터 ──
        self.declare_parameter("input_topic", "/obstacle/ground_point")
        self.declare_parameter("goal_topic", "/goal_pose")
        self.declare_parameter("goal_frame", "map")
        self.declare_parameter("move_threshold", 0.15)   # [m]
        self.declare_parameter("face_goal", True)

        self.goal_frame = self.get_parameter("goal_frame").value
        self.move_threshold = float(self.get_parameter("move_threshold").value)
        self.face_goal = bool(self.get_parameter("face_goal").value)
        input_topic = self.get_parameter("input_topic").value
        goal_topic = self.get_parameter("goal_topic").value

        # 마지막으로 발행한 목표 (중복 발행 방지용)
        self.last_goal = None  # (x, y)

        # ── 입력: 파트2의 world 절대좌표 ──
        self.create_subscription(Point, input_topic, self.on_ground_point, 10)

        # ── 출력: Nav2 목표. RViz "2D Goal Pose" 와 동일하게 latched(TRANSIENT_LOCAL) ──
        goal_qos = QoSProfile(depth=1)
        goal_qos.durability = QoSDurabilityPolicy.TRANSIENT_LOCAL
        goal_qos.reliability = QoSReliabilityPolicy.RELIABLE
        self.pub_goal = self.create_publisher(PoseStamped, goal_topic, goal_qos)

        self.get_logger().info(
            f"goal_relay_node 시작 | 입력={input_topic} → 출력={goal_topic} "
            f"(frame={self.goal_frame}, 변화기준={self.move_threshold}m)"
        )

    def on_ground_point(self, msg: Point):
        x, y = float(msg.x), float(msg.y)

        # 목표가 충분히 바뀌었을 때만 새 목표로 발행
        if self.last_goal is not None:
            dx = x - self.last_goal[0]
            dy = y - self.last_goal[1]
            if math.hypot(dx, dy) < self.move_threshold:
                return  # 사실상 같은 목표 → 무시

        goal = PoseStamped()
        goal.header.stamp = self.get_clock().now().to_msg()
        goal.header.frame_id = self.goal_frame
        goal.pose.position.x = x
        goal.pose.position.y = y
        goal.pose.position.z = 0.0

        # yaw 설정: 목표를 향하도록(원점 기준 방향) 또는 정면(0)
        if self.face_goal and (x != 0.0 or y != 0.0):
            yaw = math.atan2(y, x)
        else:
            yaw = 0.0
        goal.pose.orientation.z = math.sin(yaw / 2.0)
        goal.pose.orientation.w = math.cos(yaw / 2.0)

        self.pub_goal.publish(goal)
        self.last_goal = (x, y)
        self.get_logger().info(
            f"새 목표 발행 → /goal_pose (map: x={x:+.2f}, y={y:+.2f}, yaw={math.degrees(yaw):+.1f}°)"
        )


def main():
    rclpy.init()
    node = GoalRelayNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
