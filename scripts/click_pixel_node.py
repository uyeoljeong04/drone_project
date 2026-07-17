"""
click_pixel_node.py — 데모/테스트용 탐지 입력 노드 (옵션)
==========================================================
드론 카메라 영상(/camera/image_raw)을 창에 띄우고,
마우스로 클릭한 픽셀을 /detection/pixel 로 발행한다.
→ obstacle_locator_node 가 이 픽셀을 받아 거리·방위각 계산.

"카메라 화면에서 갯끈풀(빨간 기둥)을 클릭하면 로봇이 찾아간다" 데모용.
나중에 파트1 YOLO 탐지가 연동되면 이 노드 자리에 탐지 결과가 들어온다.

입력:  /camera/image_raw (sensor_msgs/Image)
출력:  /detection/pixel  (geometry_msgs/Point, x=u, y=v)

사용: python3 click_pixel_node.py   (q 키로 종료)

담당: 정우열 · 프로젝트: 스마트해운물류 x ICT 멘토링 (갯끈풀 탐지, 2번 파트)
"""

import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Point
from cv_bridge import CvBridge

WINDOW = "Drone Camera - click target"


class ClickPixelNode(Node):
    def __init__(self):
        super().__init__("click_pixel_node")
        self.bridge = CvBridge()
        self.last_click = None  # (u, v) 마지막 클릭 위치

        self.create_subscription(Image, "/camera/image_raw", self.on_image, 10)
        self.pub_pixel = self.create_publisher(Point, "/detection/pixel", 10)

        cv2.namedWindow(WINDOW)
        cv2.setMouseCallback(WINDOW, self.on_mouse)
        self.get_logger().info("영상 창에서 목표를 클릭하세요. (q = 종료)")

    def on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.last_click = (x, y)
            msg = Point()
            msg.x, msg.y, msg.z = float(x), float(y), 0.0
            self.pub_pixel.publish(msg)
            self.get_logger().info(f"클릭 픽셀 ({x}, {y}) → /detection/pixel 발행")

    def on_image(self, msg: Image):
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
