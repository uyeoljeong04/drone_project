"""
obstacle_locator_node.py — 2번 파트 Week 3 산출물 (병합본)
=========================================================
Week 2의 pixel_to_ground 수식을 실시간 ROS2 노드로 만든 것.
지난 채팅에서 만든 버전(파라미터 기반 입력 · TF 폴백 · 독립 실행)을 기반으로,
이번 주 핵심인 "드론 기준 → 로봇 기준 상대좌표 변환"을 추가했다.

동작 모드
  - use_tf2:=false (기본) : 카메라/로봇 모두 하드코딩된 pose 사용 (검증용)
  - use_tf2:=true         : world_frame → camera_frame, world_frame → robot_frame
                             TF를 각각 실시간 조회. 조회 실패 시 그 대상만
                             하드코딩 값으로 자동 폴백 + 경고 로그 (기존 폴백 방식 유지)

입력
  - /camera/camera_info   (sensor_msgs/CameraInfo)  → K 행렬 (수신 전에는 하드코딩 K_SIM 사용)
  - /detection/pixel      (geometry_msgs/Point)     → 탐지 픽셀 (x=u, y=v)
    (토픽이 안 들어오면 detect_pixel 파라미터 기본값을 계속 사용 — 4주차 실제 탐지기 연동 미리보기)

출력 (드론 기준, 기존 유지)
  - /obstacle/range_bearing (geometry_msgs/Point)   → x=거리[m], y=방위각[deg], z=0
  - /obstacle/ground_point  (geometry_msgs/Point)   → x=X, y=Y, z=0 (지면좌표, world 기준)

출력 (로봇 기준, 이번 주 신규 — 3번 담당자와 최종 메시지 타입/단위 합의 필요)
  - /obstacle/range_bearing_robot (geometry_msgs/Point) → x=거리[m], y=방위각[deg], z=0
                                                            (로봇 base_link 기준)
  - /obstacle/ground_point_robot  (geometry_msgs/Point) → x=X, y=Y, z=0 (로봇 기준 상대좌표)

핵심 포인트 (멘토님 지적사항):
  드론과 로봇은 world 상에서 서로 다른 위치에 있으므로, 드론이 계산한 목표 위치(P)를
  그대로 로봇에게 주면 안 된다. P를 로봇의 pose(C_robot, R_robot) 기준으로 다시 투영해야
  "로봇 입장에서 몇 m, 몇 도"가 나온다. 로봇 프레임은 실제 헤딩(IMU/EKF로 갱신)을
  반영하므로, 이 변환 결과의 방위각은 이미 헤딩 보정이 끝난 값이다.

사용 예:
  python3 obstacle_locator_node.py
  python3 obstacle_locator_node.py --ros-args -p detect_pixel:="[320.0, 19.0]"
  python3 obstacle_locator_node.py --ros-args -p use_tf2:=true -p detect_pixel:="[320.0, 19.0]"

담당: 정우열 · 프로젝트: 스마트해운물류 x ICT 멘토링 (갯끈풀 탐지, 2번 파트)
"""

import numpy as np
import rclpy
import rclpy.time
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo
from geometry_msgs.msg import Point, TransformStamped

try:
    from tf2_ros import Buffer, TransformListener, TransformBroadcaster
    TF2_AVAILABLE = True
except ImportError:
    TF2_AVAILABLE = False


# ─────────────────────────────────────────────────────────────
# 2주차 pixel_to_ground.py 와 동일한 수식 (자립 실행을 위해 내장)
# 필요하면 `from pixel_to_ground import pixel_to_ground` 로 교체해도 된다.
# ─────────────────────────────────────────────────────────────
def pixel_to_ground(u, v, K, R, C):
    K = np.asarray(K, dtype=float)
    R = np.asarray(R, dtype=float)
    C = np.asarray(C, dtype=float)

    pixel = np.array([u, v, 1.0])       # 동차좌표
    d_cam = np.linalg.inv(K) @ pixel    # 픽셀 → 카메라 광선 방향
    d_world = R @ d_cam                 # 월드 좌표계 방향

    if d_world[2] >= 0:
        raise ValueError(
            "광선이 지면을 향하지 않습니다 (d_world_z >= 0). "
            "카메라 자세 R 또는 광학 프레임 축 방향을 확인하세요."
        )

    t = -C[2] / d_world[2]              # 지면(z=0) 교차 비율
    P = C + t * d_world                 # 지면 위 실제 좌표

    dx, dy = P[0] - C[0], P[1] - C[1]
    distance = np.sqrt(dx ** 2 + dy ** 2)
    bearing = np.arctan2(dy, dx)
    return distance, bearing, P


def quaternion_to_matrix(qx, qy, qz, qw):
    """쿼터니언 → 3x3 회전행렬 (tf2 조회 결과를 R로 변환할 때 사용)."""
    n = qx * qx + qy * qy + qz * qz + qw * qw
    if n < 1e-12:
        return np.eye(3)
    s = 2.0 / n
    xx, yy, zz = qx * qx * s, qy * qy * s, qz * qz * s
    xy, xz, yz = qx * qy * s, qx * qz * s, qy * qz * s
    wx, wy, wz = qw * qx * s, qw * qy * s, qw * qz * s
    return np.array([
        [1.0 - (yy + zz),       xy - wz,        xz + wy],
        [     xy + wz,     1.0 - (xx + zz),     yz - wx],
        [     xz - wy,          yz + wx,   1.0 - (xx + yy)],
    ])


# ── 검증용 하드코딩 값 (2주차 validate_projection.py 와 동일) ──────────
# world 파일 카메라 pose: 0 0 5  0 1.5708 0  (원점 위 5m, 똑바로 아래)
K_SIM = np.array([[554.38,   0.0, 320.5],
                   [  0.0, 554.38, 240.5],
                   [  0.0,   0.0,   1.0]])
C_HARDCODED = np.array([0.0, 0.0, 5.0])
R_HARDCODED = np.array([[ 0.0, -1.0,  0.0],
                         [-1.0,  0.0,  0.0],
                         [ 0.0,  0.0, -1.0]])

# ── 로봇 검증용 하드코딩 값 (이번 주 신규) ─────────────────────────────
# TF가 아직 없을 때의 폴백: 실제 로봇 스폰 위치/자세로 바꿔서 테스트할 것.
# 회전 없음(단위행렬) = 로봇이 world +X 방향을 정면으로 보고 있다고 가정.
ROBOT_POS_HARDCODED = np.array([2.0, 0.0, 0.0])
ROBOT_ROT_HARDCODED = np.eye(3)


class ObstacleLocatorNode(Node):
    def __init__(self):
        super().__init__("obstacle_locator_node")

        # ── 파라미터 ──
        self.declare_parameter("use_tf2", False)
        self.declare_parameter("detect_pixel", [320.0, 240.0])
        self.declare_parameter("camera_frame", "camera_optical")
        self.declare_parameter("robot_frame", "robot/base_link")
        self.declare_parameter("world_frame", "world")
        self.declare_parameter("rate_hz", 2.0)

        self.use_tf2 = bool(self.get_parameter("use_tf2").value)
        self.camera_frame = self.get_parameter("camera_frame").value
        self.robot_frame = self.get_parameter("robot_frame").value
        self.world_frame = self.get_parameter("world_frame").value
        rate_hz = float(self.get_parameter("rate_hz").value)

        px = self.get_parameter("detect_pixel").value
        self.current_pixel = (float(px[0]), float(px[1]))

        # ── 카메라 K: camera_info 수신 전엔 하드코딩 값으로 시작 ──
        self.K = K_SIM.copy()
        self.create_subscription(CameraInfo, "/camera/camera_info", self.on_camera_info, 10)

        # ── 탐지 픽셀 입력 (4주차에서 실제 탐지기 출력으로 교체될 자리) ──
        self.create_subscription(Point, "/detection/pixel", self.on_detection_pixel, 10)

        # ── 결과 발행: 드론 기준(기존) + 로봇 기준(신규) ──
        self.pub_range_bearing = self.create_publisher(Point, "/obstacle/range_bearing", 10)
        self.pub_ground_point = self.create_publisher(Point, "/obstacle/ground_point", 10)
        self.pub_range_bearing_robot = self.create_publisher(Point, "/obstacle/range_bearing_robot", 10)
        self.pub_ground_point_robot = self.create_publisher(Point, "/obstacle/ground_point_robot", 10)

        # ── 목표 지점(P)을 TF로도 발행 (RViz 시각 확인용, use_tf2 설정과 무관하게 항상 동작) ──
        self.tf_broadcaster = TransformBroadcaster(self) if TF2_AVAILABLE else None

        # ── tf2 (use_tf2:=true 일 때만 사용) ──
        self.tf_buffer = None
        if self.use_tf2:
            if not TF2_AVAILABLE:
                self.get_logger().warn("tf2_ros 를 불러올 수 없습니다 → 하드코딩 모드로 전환합니다.")
                self.use_tf2 = False
            else:
                self.tf_buffer = Buffer()
                self._tf_listener = TransformListener(self.tf_buffer, self)  # noqa: F841 (참조 유지용)

        mode_str = "tf2" if self.use_tf2 else "하드코딩"
        self.get_logger().info(
            f"obstacle_locator_node 시작 (모드={mode_str}, rate={rate_hz}Hz, "
            f"초기 detect_pixel={self.current_pixel})"
        )

        self.timer = self.create_timer(1.0 / rate_hz, self.on_timer)

    def on_camera_info(self, msg):
        self.K = np.array(msg.k, dtype=float).reshape(3, 3)

    def on_detection_pixel(self, msg):
        self.current_pixel = (float(msg.x), float(msg.y))
        self.get_logger().info(f"/detection/pixel 수신 → 픽셀({msg.x:.1f}, {msg.y:.1f})")

    def _lookup_pose(self, frame_id, fallback_R, fallback_C, label):
        """world_frame → frame_id TF를 조회해 (R, C)를 반환. 실패 시 폴백 + 경고."""
        if not self.use_tf2:
            return fallback_R, fallback_C

        try:
            tf = self.tf_buffer.lookup_transform(
                self.world_frame, frame_id, rclpy.time.Time()
            )
        except Exception as e:
            self.get_logger().warn(f"{label} tf2 조회 실패({frame_id}) → 폴백 ({e})")
            return fallback_R, fallback_C

        t = tf.transform.translation
        q = tf.transform.rotation
        C = np.array([t.x, t.y, t.z])
        R = quaternion_to_matrix(q.x, q.y, q.z, q.w)
        return R, C

    def get_camera_R_C(self):
        return self._lookup_pose(self.camera_frame, R_HARDCODED, C_HARDCODED, "카메라")

    def get_robot_R_C(self):
        return self._lookup_pose(self.robot_frame, ROBOT_ROT_HARDCODED, ROBOT_POS_HARDCODED, "로봇")

    def on_timer(self):
        u, v = self.current_pixel
        R_cam, C_cam = self.get_camera_R_C()

        # 1) 픽셀 → world 기준 지면좌표 (드론 기준 거리/방위각)
        try:
            distance, bearing, P = pixel_to_ground(u, v, self.K, R_cam, C_cam)
        except ValueError as e:
            self.get_logger().warn(f"투영 실패: {e}")
            return

        bearing_deg = float(np.degrees(bearing))

        # RViz 시각 확인용: 목표 지점 P를 world 기준 TF("target")로도 발행
        if self.tf_broadcaster is not None:
            t_msg = TransformStamped()
            t_msg.header.stamp = self.get_clock().now().to_msg()
            t_msg.header.frame_id = self.world_frame
            t_msg.child_frame_id = "target"
            t_msg.transform.translation.x = float(P[0])
            t_msg.transform.translation.y = float(P[1])
            t_msg.transform.translation.z = float(P[2])
            t_msg.transform.rotation.w = 1.0  # 회전 없음
            self.tf_broadcaster.sendTransform(t_msg)

        # 2) world 기준 P를 로봇 pose(R_robot, C_robot) 기준으로 재투영
        #    = 드론이 본 좌표를 "로봇 입장의 몇 m, 몇 도"로 변환하는 이번 주 핵심 단계
        R_robot, C_robot = self.get_robot_R_C()
        P_rel = R_robot.T @ (P - C_robot)   # 로봇 기준 상대좌표 (x=전방, y=좌측 등은 R_robot 정의에 따름)
        rel_distance = float(np.hypot(P_rel[0], P_rel[1]))
        rel_bearing_deg = float(np.degrees(np.arctan2(P_rel[1], P_rel[0])))

        self.get_logger().info(
            f"픽셀({u:.0f},{v:.0f}) → 지면(X={P[0]:+.2f}, Y={P[1]:+.2f}) "
            f"| [드론기준] 거리={distance:.2f}m 방위={bearing_deg:+.1f}°  "
            f"| [로봇기준] 거리={rel_distance:.2f}m 방위={rel_bearing_deg:+.1f}°"
        )

        # ── 발행: 드론 기준 (기존) ──
        rb = Point()
        rb.x, rb.y, rb.z = float(distance), bearing_deg, 0.0
        self.pub_range_bearing.publish(rb)

        gp = Point()
        gp.x, gp.y, gp.z = float(P[0]), float(P[1]), 0.0
        self.pub_ground_point.publish(gp)

        # ── 발행: 로봇 기준 (신규) ──
        rb_r = Point()
        rb_r.x, rb_r.y, rb_r.z = rel_distance, rel_bearing_deg, 0.0
        self.pub_range_bearing_robot.publish(rb_r)

        gp_r = Point()
        gp_r.x, gp_r.y, gp_r.z = float(P_rel[0]), float(P_rel[1]), 0.0
        self.pub_ground_point_robot.publish(gp_r)


def main():
    rclpy.init()
    node = ObstacleLocatorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()