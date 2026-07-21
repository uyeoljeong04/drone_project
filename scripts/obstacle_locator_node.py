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

[Week5 / Phase 3 변경] 탐지를 '상태'에서 '사건'으로 재구조화
  이전에는 마지막 탐지 픽셀을 저장해 두고 매 주기 '최신 TF'로 다시 투영했다.
  드론이 고정일 때는 무해했지만, 드론이 움직이면
  "3초 전에 촬영한 픽셀을 지금 위치에서 본 것처럼" 계산하게 되어
  목표 좌표가 드론 속도와 1:1로 흘러가 버린다 (Week5 실측 0.158 m 오차).

  → 탐지가 들어오면 그 영상이 촬영된 시각의 드론 pose 로 **딱 한 번** 투영하고,
    그렇게 확정된 map 좌표는 이후 드론이 어디로 가든 변하지 않는다.
    갯끈풀은 움직이지 않는 고정 개체이므로 이것이 물리적으로 옳다.
  → 단, 로봇 기준 상대좌표는 로봇이 계속 움직이므로 매 주기 재계산한다.

입력
  - /drone/camera_info    (sensor_msgs/CameraInfo)      → K 행렬 (수신 전에는 K_SIM 사용)
  - /detection/pixel      (geometry_msgs/PointStamped)  → 탐지 픽셀 (point.x=u, point.y=v)
    ※ header.stamp 에 반드시 '그 픽셀이 찍힌 영상의 시각'을 실어야 한다.

출력 (드론 기준 — 촬영 시점 기준으로 고정)
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
from geometry_msgs.msg import Point, PointStamped, TransformStamped

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


# ── 검증용 하드코딩 값 ──────────────────────────────────────────────
# ※ use_tf2:=false 일 때만 쓰인다. use_tf2:=true 에서는 더 이상 폴백으로
#   쓰이지 않는다 (_lookup_pose 주석 참고).
# ※ 고도 5.0 → 8.0 수정 (Week5):
#   world 의 카메라가 5m → 8m 로 바뀌었는데 이 상수만 5.0 으로 남아 있었다.
#   Week2 검증 결과(5m 기준)를 재현하려면 이 값을 5.0 으로 되돌릴 것.
K_SIM = np.array([[554.38,   0.0, 320.5],
                   [  0.0, 554.38, 240.5],
                   [  0.0,   0.0,   1.0]])
C_HARDCODED = np.array([0.0, 0.0, 8.0])
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
        # [Week5/Phase3] 탐지 픽셀에 실린 영상 시각으로 TF를 조회할지 여부.
        #   true  (기본) : 영상이 촬영된 시각의 드론 pose 로 투영  ← 올바른 동작
        #   false        : '지금'의 최신 TF 로 투영  ← Phase3 이전의 동작
        # 수정 전/후 오차를 비교 측정할 때 이 파라미터만 바꾸면 된다.
        self.declare_parameter("use_detection_stamp", True)

        self.use_tf2 = bool(self.get_parameter("use_tf2").value)
        self.camera_frame = self.get_parameter("camera_frame").value
        self.robot_frame = self.get_parameter("robot_frame").value
        self.world_frame = self.get_parameter("world_frame").value
        rate_hz = float(self.get_parameter("rate_hz").value)
        self.use_detection_stamp = bool(self.get_parameter("use_detection_stamp").value)

        # ── [Week5/Phase3] 탐지는 '상태'가 아니라 '사건' ──
        #   pending : 아직 투영하지 못한 탐지 (u, v, stamp, 접수시각)
        #   target  : 투영이 끝나 확정된 목표. 한 번 정해지면 드론이 어디로 가든 불변.
        px = self.get_parameter("detect_pixel").value
        self.pending = (float(px[0]), float(px[1]), None, None)
        self.target = None

        # ── 카메라 K: camera_info 수신 전엔 하드코딩 값으로 시작 ──
        self.K = K_SIM.copy()
        self.create_subscription(CameraInfo, "/drone/camera_info", self.on_camera_info, 10)

        # ── 탐지 픽셀 입력 ──
        # [Week5/Phase3] Point → PointStamped 로 변경.
        #   header.stamp 에 '그 픽셀이 찍힌 영상의 시각'이 들어온다.
        #   발행하는 쪽(click_pixel_node / 탐지기)이 영상 헤더를 그대로 복사해야 한다.
        self.create_subscription(PointStamped, "/detection/pixel",
                                 self.on_detection_pixel, 10)

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
            f"초기 detect_pixel=({self.pending[0]:.0f}, {self.pending[1]:.0f}), "
            f"영상시각 사용={self.use_detection_stamp})"
        )

        self.timer = self.create_timer(1.0 / rate_hz, self.on_timer)

    def on_camera_info(self, msg):
        self.K = np.array(msg.k, dtype=float).reshape(3, 3)

    def on_detection_pixel(self, msg: PointStamped):
        """탐지 1건 접수. 실제 투영은 on_timer 에서 1회만 수행한다."""
        self.pending = (float(msg.point.x), float(msg.point.y),
                        msg.header.stamp, self.get_clock().now())
        self.get_logger().info(
            f"탐지 수신 → 픽셀({msg.point.x:.1f}, {msg.point.y:.1f}) "
            f"@ 영상시각 {msg.header.stamp.sec}.{msg.header.stamp.nanosec // 1000000:03d}"
        )

    def _project_pending(self):
        """대기 중인 탐지를 '그 영상이 찍힌 시각의 드론 pose' 로 1회 투영해 확정한다."""
        if self.pending is None:
            return

        u, v, stamp, recv_time = self.pending

        # use_detection_stamp:=false 면 예전처럼 최신 TF 를 쓴다 (오차 비교 실험용)
        lookup_stamp = None
        if stamp is not None and self.use_detection_stamp:
            lookup_stamp = rclpy.time.Time.from_msg(stamp)

        cam = self.get_camera_R_C(lookup_stamp)
        if cam is None:
            # TF 가 아직 안 들어왔을 수 있으므로 다음 주기에 재시도.
            # 다만 너무 오래된 탐지는 버린다 (TF 버퍼에서 사라지면 영영 못 씀).
            if recv_time is not None:
                age = (self.get_clock().now() - recv_time).nanoseconds * 1e-9
                if age > 3.0:
                    self.get_logger().error(
                        f"탐지 픽셀({u:.0f},{v:.0f}) 투영 실패 — 해당 시각의 TF 없음. 폐기."
                    )
                    self.pending = None
            return
        R_cam, C_cam = cam

        try:
            distance, bearing, P = pixel_to_ground(u, v, self.K, R_cam, C_cam)
        except ValueError as e:
            self.get_logger().warn(f"투영 실패: {e}")
            self.pending = None
            return

        self.target = {
            "P": P,
            "uv": (u, v),
            "drone_dist": float(distance),
            "drone_bearing_deg": float(np.degrees(bearing)),
            "drone_C": C_cam.copy(),
        }
        self.pending = None

        mode = "영상시각" if lookup_stamp is not None else "최신TF"
        self.get_logger().info(
            f"★ 목표 확정 [{mode}] 픽셀({u:.0f},{v:.0f}) → map({P[0]:+.3f}, {P[1]:+.3f}) "
            f"| 촬영 시점 드론 위치 ({C_cam[0]:+.2f}, {C_cam[1]:+.2f}, {C_cam[2]:+.2f})"
        )

    def _lookup_pose(self, frame_id, fallback_R, fallback_C, label, stamp=None):
        """
        world_frame → frame_id TF를 조회해 (R, C)를 반환.

        stamp=None      → 최신 TF (로봇처럼 '지금' 상태가 필요한 대상)
        stamp=<Time>    → 그 시각의 TF (카메라처럼 '촬영 순간' 상태가 필요한 대상)

        use_tf2:=false  → 하드코딩 값 사용 (Week2~3 검증 모드)
        use_tf2:=true   → TF 조회. 실패하면 None 을 반환한다.

        ※ Week5 변경: 예전에는 TF 조회에 실패하면 하드코딩 값으로 폴백했다.
          드론이 고정이던 시절엔 그 상수가 우연히 맞았지만, 드론이 움직이는
          지금은 어떤 상수를 넣어도 틀린 값이다. 그런데 계산은 계속 성공하고
          로봇도 정상적으로 움직여서, 조용히 엉뚱한 곳으로 가는
          '가장 찾기 어려운 종류의 버그'가 된다.
          → 폴백을 없애고 그 주기를 건너뛴다. 출력이 멈추면 즉시 알아챌 수 있다.
        """
        if not self.use_tf2:
            return fallback_R, fallback_C

        try:
            tf = self.tf_buffer.lookup_transform(
                self.world_frame, frame_id,
                stamp if stamp is not None else rclpy.time.Time(),
            )
        except Exception as e:
            self.get_logger().error(
                f"{label} TF 조회 실패 ({self.world_frame} → {frame_id}): {e}\n"
                f"  → 틀린 좌표를 내지 않기 위해 이번 주기 계산을 건너뜁니다.",
                throttle_duration_sec=2.0,
            )
            return None

        t = tf.transform.translation
        q = tf.transform.rotation
        C = np.array([t.x, t.y, t.z])
        R = quaternion_to_matrix(q.x, q.y, q.z, q.w)
        return R, C

    def get_camera_R_C(self, stamp=None):
        return self._lookup_pose(self.camera_frame, R_HARDCODED, C_HARDCODED,
                                 "카메라", stamp)

    def get_robot_R_C(self):
        # 로봇은 '지금' 어디를 보고 있는지가 중요하므로 항상 최신 TF
        return self._lookup_pose(self.robot_frame, ROBOT_ROT_HARDCODED,
                                 ROBOT_POS_HARDCODED, "로봇")

    def on_timer(self):
        # ── 1) 새 탐지가 있으면 '그 영상 시각의 pose' 로 1회만 투영해 확정 ──
        self._project_pending()

        if self.target is None:
            return          # 아직 확정된 목표 없음

        # 확정된 값들 — 드론이 이후 어디로 움직여도 변하지 않는다.
        P = self.target["P"]
        u, v = self.target["uv"]
        distance = self.target["drone_dist"]
        bearing_deg = self.target["drone_bearing_deg"]

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

        # ── 2) 확정된 map 좌표 P 를 '현재' 로봇 pose 기준으로 매 주기 재계산 ──
        #   목표(P)는 고정이지만 로봇은 계속 움직이므로, 로봇 기준 거리·방위각은
        #   매번 새로 구해야 한다. 로봇이 회전하는 동안 방위각이 실시간으로
        #   갱신되어야 target_follower 가 제대로 조향할 수 있다. (Week4 검증 내용)
        rob = self.get_robot_R_C()
        if rob is None:
            return          # 로봇 TF 없음 → 상대좌표를 낼 수 없으므로 생략
        R_robot, C_robot = rob

        P_rel = R_robot.T @ (P - C_robot)   # 로봇 기준 상대좌표 (x=전방, y=좌측 등은 R_robot 정의에 따름)
        rel_distance = float(np.hypot(P_rel[0], P_rel[1]))
        rel_bearing_deg = float(np.degrees(np.arctan2(P_rel[1], P_rel[0])))

        self.get_logger().info(
            f"목표 map(X={P[0]:+.2f}, Y={P[1]:+.2f}) [고정] "
            f"| [드론기준·촬영시점] 거리={distance:.2f}m 방위={bearing_deg:+.1f}°  "
            f"| [로봇기준·실시간] 거리={rel_distance:.2f}m 방위={rel_bearing_deg:+.1f}°",
            throttle_duration_sec=1.0,
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