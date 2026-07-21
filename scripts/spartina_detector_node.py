"""
spartina_detector_node.py — 시뮬용 자동 탐지기 (색 기반)
========================================================
드론 영상에서 목표(빨간 원기둥)를 HSV 색 검출로 자동으로 찾아
/detection/pixel 로 발행한다. click_pixel_node 를 대체한다.

[역할 구분 — 중요]
  이 노드는 **파이프라인 검증용**이다. 실제 갯끈풀 탐지가 아니다.

  ┌ 색 검출기 (이 파일) ─ 시뮬 빨간 기둥 ─ "탐지→좌표변환→주행" 이 옳게 도는가
  └ YOLO (파트1 담당)  ─ 실제 갯벌 사진 ─ "갯끈풀을 얼마나 잘 찾는가"

  둘은 검증 대상이 다르므로 별개로 보고한다.
  시뮬 월드엔 실제 갯끈풀이 없고, 실사진으로 학습한 YOLO 는 빨간 원기둥을
  찾지 못한다(도메인 불일치). 따라서 시뮬 안에서는 색 검출로 파이프라인을 돌린다.

[왜 만들었나 — 측정 정밀도]
  기존 click_pixel_node 는 사람이 화면을 보고 마우스를 눌렀다.
  화면을 본 시점과 클릭이 등록된 시점 사이에 사람의 반응 시간(약 0.15~0.25초)이
  끼어드는데, 드론이 3 m/s 로 날면 그것만으로 0.5~0.75 m 오차가 생긴다.
  이 오차는 시스템 문제가 아니라 '측정 방법'의 한계라 코드로는 못 없앤다.
  → 사람을 빼면 사라진다. 검출에 사용한 프레임의 header 를 그대로 싣기 때문.

╔══════════════════════════════════════════════════════════════════╗
║ 인터페이스 규약 — 파트1(YOLO)이 이 노드를 대체할 때 지켜야 할 것  ║
╠══════════════════════════════════════════════════════════════════╣
║ 토픽 : /detection/pixel                                          ║
║ 타입 : geometry_msgs/PointStamped                                ║
║                                                                  ║
║   header.stamp    ★ 탐지에 사용한 **영상의 header.stamp 를 그대로**║
║                     복사할 것. 현재 시각(now())을 쓰면 안 된다.   ║
║                     드론이 움직이면 이 값이 좌표 정확도를 좌우한다.║
║   header.frame_id   영상의 frame_id (drone_camera_optical)        ║
║   point.x         = u  (픽셀 열, 0~639)                           ║
║   point.y         = v  (픽셀 행, 0~479)                           ║
║   point.z         = 0  (미사용)                                   ║
║                                                                  ║
║ 어느 픽셀을 보낼 것인가:                                          ║
║   바운딩박스의 **중심**을 보낸다.                                 ║
║   수직 하방 촬영이므로 식물의 윗면이 보이며, 투영기는 지면(z=0)을 ║
║   가정하므로 식물 높이만큼의 체계적 편향이 남는다.                ║
║   (갯끈풀 높이 0.3~1 m → 화면 가장자리에서 수십 cm)               ║
║   이는 알려진 한계이며 별도로 다룬다.                             ║
╚══════════════════════════════════════════════════════════════════╝

입력:  /drone/image_raw   (sensor_msgs/Image)
출력:  /detection/pixel   (geometry_msgs/PointStamped)
       /detection/debug_image (sensor_msgs/Image) — 검출 결과 표시용

사용:
  python3 spartina_detector_node.py --ros-args -p use_sim_time:=true
  python3 spartina_detector_node.py --ros-args -p show_window:=true   # 창 띄우기

담당: 정우열 · 스마트해운물류 x ICT 멘토링 (갯끈풀 탐지, 2번 파트)
"""

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import PointStamped
from cv_bridge import CvBridge

WINDOW = "Spartina Detector"


class SpartinaDetectorNode(Node):
    def __init__(self):
        super().__init__("spartina_detector_node")

        # ── 파라미터 ──
        # 빨강은 HSV 색상환에서 0도 근처(0~180 스케일에서 0 과 180 양쪽)에 걸쳐 있어
        # 구간을 두 개로 나눠서 잡아야 한다.
        self.declare_parameter("h_low1", 0)      # 빨강 구간 1 (0 쪽)
        self.declare_parameter("h_high1", 10)
        self.declare_parameter("h_low2", 160)    # 빨강 구간 2 (180 쪽)
        self.declare_parameter("h_high2", 180)
        self.declare_parameter("s_min", 100)     # 채도 하한 (회색 장애물 배제)
        self.declare_parameter("v_min", 60)      # 명도 하한 (그림자 배제)
        self.declare_parameter("min_area_px", 60)     # 이보다 작은 덩어리는 노이즈로 무시
        self.declare_parameter("publish_rate_hz", 5.0)  # 발행 상한 (0 이면 매 프레임)
        self.declare_parameter("show_window", False)

        g = lambda n: self.get_parameter(n).value
        self.h1 = (int(g("h_low1")), int(g("h_high1")))
        self.h2 = (int(g("h_low2")), int(g("h_high2")))
        self.s_min = int(g("s_min"))
        self.v_min = int(g("v_min"))
        self.min_area = int(g("min_area_px"))
        rate = float(g("publish_rate_hz"))
        self.min_interval = (1.0 / rate) if rate > 0 else 0.0
        self.show_window = bool(g("show_window"))

        self.bridge = CvBridge()
        self.last_pub_time = None
        self.n_detect = 0
        self.n_frame = 0

        self.create_subscription(Image, "/drone/image_raw", self.on_image, 10)
        self.pub_pixel = self.create_publisher(PointStamped, "/detection/pixel", 10)
        self.pub_debug = self.create_publisher(Image, "/detection/debug_image", 10)

        if self.show_window:
            cv2.namedWindow(WINDOW)

        self.get_logger().info(
            f"spartina_detector 시작 | 빨강 H {self.h1}·{self.h2}, S≥{self.s_min}, "
            f"V≥{self.v_min}, 최소면적 {self.min_area}px, 발행 {rate}Hz"
        )

    # ─────────────────────────────────────────────
    def _find_target(self, bgr):
        """가장 큰 빨간 덩어리의 (중심픽셀, 면적, 윤곽). 없으면 None."""
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

        m1 = cv2.inRange(hsv, (self.h1[0], self.s_min, self.v_min),
                              (self.h1[1], 255, 255))
        m2 = cv2.inRange(hsv, (self.h2[0], self.s_min, self.v_min),
                              (self.h2[1], 255, 255))
        mask = cv2.bitwise_or(m1, m2)

        # 잡티 제거 → 구멍 메우기
        k = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        c = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(c)
        if area < self.min_area:
            return None

        M = cv2.moments(c)
        if M["m00"] == 0:
            return None
        u = M["m10"] / M["m00"]
        v = M["m01"] / M["m00"]
        return (u, v), area, c

    # ─────────────────────────────────────────────
    def on_image(self, msg: Image):
        self.n_frame += 1
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        found = self._find_target(frame)

        if found is not None:
            (u, v), area, contour = found
            self.n_detect += 1

            # ── 발행 속도 제한 ──
            now = self.get_clock().now()
            ok = True
            if self.min_interval > 0 and self.last_pub_time is not None:
                dt = (now - self.last_pub_time).nanoseconds * 1e-9
                ok = dt >= self.min_interval

            if ok:
                out = PointStamped()
                # ★ 규약: '지금'이 아니라 이 영상의 header 를 그대로 쓴다.
                out.header = msg.header
                out.point.x, out.point.y, out.point.z = float(u), float(v), 0.0
                self.pub_pixel.publish(out)
                self.last_pub_time = now

            # ── 디버그 영상 ──
            cv2.drawContours(frame, [contour], -1, (0, 255, 0), 2)
            cv2.drawMarker(frame, (int(round(u)), int(round(v))), (255, 0, 255),
                           cv2.MARKER_CROSS, 20, 2)
            cv2.putText(frame, f"({u:.1f}, {v:.1f})  area={int(area)}",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
        else:
            cv2.putText(frame, "no target", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        self.pub_debug.publish(self.bridge.cv2_to_imgmsg(frame, encoding="bgr8"))

        if self.show_window:
            cv2.imshow(WINDOW, frame)
            cv2.waitKey(1)

        # 100프레임마다 검출률 보고
        if self.n_frame % 100 == 0:
            self.get_logger().info(
                f"검출률 {self.n_detect}/{self.n_frame} "
                f"({100.0 * self.n_detect / self.n_frame:.1f}%)"
            )


def main():
    rclpy.init()
    node = SpartinaDetectorNode()
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
