"""
click_pixel_node.py — 데모/테스트용 탐지 입력 노드 (옵션)
==========================================================
드론 카메라 영상(/camera/image_raw)을 창에 띄우고,
마우스로 클릭한 픽셀을 /detection/pixel 로 발행한다.
→ obstacle_locator_node 가 이 픽셀을 받아 거리·방위각 계산.

"카메라 화면에서 갯끈풀(빨간 기둥)을 클릭하면 로봇이 찾아간다" 데모용.
나중에 파트1 YOLO 탐지가 연동되면 이 노드 자리에 탐지 결과가 들어온다.

입력:  /drone/image_raw (sensor_msgs/Image)
출력:  /detection/pixel (geometry_msgs/PointStamped, point.x=u, point.y=v)

[Week5 / Phase 3 변경] Point → PointStamped
  드론이 움직이면서 "언제 찍힌 영상의 픽셀인가"가 중요해졌다.
  클릭한 픽셀이 속한 **영상 프레임의 header 를 그대로 복사**해서 함께 보낸다.
  obstacle_locator 는 이 시각의 드론 pose 로 투영하므로,
  화면에 표시되기까지의 지연이나 사람의 반응 시간이 있어도 좌표가 밀리지 않는다.
  (실측: 이 처리가 없으면 0.8 m/s 비행 중 약 0.16 m 오차)

  나중에 YOLO 탐지기로 교체할 때도 동일하게
  "탐지에 사용한 영상의 header.stamp"를 실어 보내야 한다.

사용: python3 click_pixel_node.py   (q 키로 종료)

담당: 정우열 · 프로젝트: 스마트해운물류 x ICT 멘토링 (갯끈풀 탐지, 2번 파트)
"""

import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import PointStamped
from cv_bridge import CvBridge

WINDOW = "Drone Camera - click target"


class ClickPixelNode(Node):
    def __init__(self):
        super().__init__("click_pixel_node")
        self.bridge = CvBridge()
        self.last_click = None    # (u, v) 마지막 클릭 위치
        self.last_header = None   # 화면에 현재 표시 중인 영상의 header (시각 포함)

        self.create_subscription(Image, "/drone/image_raw", self.on_image, 10)
        self.pub_pixel = self.create_publisher(PointStamped, "/detection/pixel", 10)

        cv2.namedWindow(WINDOW)
        cv2.setMouseCallback(WINDOW, self.on_mouse)
        self.get_logger().info("영상 창에서 목표를 클릭하세요. (q = 종료)")

    def on_mouse(self, event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        if self.last_header is None:
            self.get_logger().warn("아직 영상을 받지 못했습니다. 잠시 후 다시 클릭하세요.")
            return

        self.last_click = (x, y)

        msg = PointStamped()
        # ★ 핵심: '지금 시각'이 아니라 **화면에 표시 중인 영상의 시각**을 싣는다.
        #   클릭은 사람이 화면을 보고 누르는 것이므로 항상 과거 프레임에 대한 것이다.
        msg.header = self.last_header
        msg.point.x, msg.point.y, msg.point.z = float(x), float(y), 0.0
        self.pub_pixel.publish(msg)

        self.get_logger().info(
            f"클릭 픽셀 ({x}, {y}) → /detection/pixel 발행 "
            f"(영상시각 {msg.header.stamp.sec}.{msg.header.stamp.nanosec // 1000000:03d})"
        )

    def on_image(self, msg: Image):
        self.last_header = msg.header      # 클릭 시 그대로 복사해 쓸 헤더
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

        # 마지막 클릭 위치 표시
        if self.last_click is not None:
            u, v = self.last_click
            cv2.drawMarker(frame, (u, v), (0, 0, 255),
                           markerType=cv2.MARKER_CROSS, markerSize=20, thickness=2)
            cv2.putText(frame, f"target ({u},{v})", (u + 10, v - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        cv2.imshow(WINDOW, frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            self.get_logger().info("q 입력 → 종료")
            rclpy.shutdown()


def main():
    rclpy.init()
    node = ClickPixelNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
