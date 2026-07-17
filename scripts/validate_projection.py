"""
validate_projection.py  —  2번 파트 Week 2 검증 도구
=====================================================
Gazebo 시뮬레이터의 드론 카메라 영상을 받아 화면에 띄우고,
영상에서 콘(목표)을 '마우스로 클릭'하면 그 픽셀을 지면 좌표로 투영해
거리·방향각·지면좌표(P)를 출력한다.

사용법:
  1) Gazebo 월드를 먼저 실행해 카메라 토픽이 나오는 상태로 둔다.
  2) python3 validate_projection.py
  3) OpenCV 창에서 콘의 중심을 클릭 → 터미널에 계산값 출력
  4) 그 계산값을, Gazebo에서 아는 콘의 실제 위치와 비교 → 오차 표 작성

전제: 카메라가 똑바로 아래를 보고, 원점 위 CAMERA_HEIGHT 높이에 있다고 가정.
      (검증이라 손으로 넣음. 실제 노드에서는 3주차에 TF로 대체.)
"""

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import cv2

from pixel_to_ground import pixel_to_ground

# ── 검증용 카메라 설정: 네 Gazebo 월드와 일치하게 맞춰라 ──────────────
# world 파일의 카메라 pose:  0 0 5  0 1.5708 0   (x y z  roll pitch yaw)
CAMERA_HEIGHT = 5.0                        # 카메라 높이 [m]
C = np.array([0.0, 0.0, CAMERA_HEIGHT])    # 카메라 월드 위치 [x, y, z]

# 카메라 자세 R (광학 프레임 → 월드).
# 이 카메라는 pitch=90°(1.5708)로 눕혀 똑바로 아래를 본다.
# 그 pose + ROS 광학 프레임 규약(z=앞, x=오른쪽, y=아래)을 합치면:
#   광학 z(앞)     → 월드 -z (아래)
#   광학 x(오른쪽) → 월드 -y
#   광학 y(아래)   → 월드 -x
R = np.array([[ 0.0, -1.0,  0.0],
              [-1.0,  0.0,  0.0],
              [ 0.0,  0.0, -1.0]])
# ─────────────────────────────────────────────────────────────────────


class ProjectionValidator(Node):
    def __init__(self):
        super().__init__("projection_validator")
        self.bridge = CvBridge()
        self.K = None          # /camera/camera_info 에서 채워짐
        self.latest = None     # 최신 영상 프레임

        self.create_subscription(Image, "/camera/image_raw",
                                 self.on_image, 10)
        self.create_subscription(CameraInfo, "/camera/camera_info",
                                 self.on_info, 10)

        cv2.namedWindow("camera")
        cv2.setMouseCallback("camera", self.on_click)
        self.get_logger().info("콘을 마우스로 클릭하면 지면 좌표를 계산합니다.")

    def on_info(self, msg):
        # CameraInfo.k 는 길이 9 배열 → 3x3 로 재구성
        self.K = np.array(msg.k, dtype=float).reshape(3, 3)

    def on_image(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        self.latest = frame
        cv2.imshow("camera", frame)
        cv2.waitKey(1)

    def on_click(self, event, u, v, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        if self.K is None:
            self.get_logger().warn("아직 K(camera_info)를 못 받았습니다.")
            return
        dist, bearing, P = pixel_to_ground(u, v, self.K, R, C)
        print("─" * 56)
        print(f" 클릭 픽셀   : (u={u}, v={v})")
        print(f" 지면 좌표 P : X={P[0]:+.3f}, Y={P[1]:+.3f}, Z={P[2]:+.3f}")
        print(f" 거리        : {dist:.3f} m")
        print(f" 방향각      : {np.degrees(bearing):+.1f}°  ({bearing:+.4f} rad)")
        print("─" * 56)


def main():
    rclpy.init()
    node = ProjectionValidator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
