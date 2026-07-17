"""
pixel_to_ground.py  —  2번 파트 Week 2 산출물
=================================================
드론 카메라 영상 속 한 픽셀 (u, v)를 지면(z=0) 위 실제 좌표로 투영해
드론으로부터의 '거리'와 '방향각(방위)'을 계산한다.

원리: 지면 평면 투영 (Ground-Plane Projection)
  ① d_cam   = K⁻¹ @ [u, v, 1]     픽셀 → 카메라 좌표계 광선 방향
  ② d_world = R  @ d_cam           카메라 기준 → 월드 기준 방향
  ③ t       = -C_z / d_world_z     광선이 지면(z=0)에 닿는 비율
  ④ P       = C + t * d_world      땅 위 실제 좌표 [X, Y, 0]
  → 거리   = sqrt(dx² + dy²),  방향각 = atan2(dy, dx)

좌표계 규약: ROS 카메라 광학 프레임 (z=앞, x=오른쪽, y=아래)

담당: 정우열  ·  프로젝트: 스마트해운물류 x ICT 멘토링 (갯끈풀 탐지)
"""

import numpy as np


# ─────────────────────────────────────────────────────────────
# 핵심 함수
# ─────────────────────────────────────────────────────────────
def pixel_to_ground(u, v, K, R, C):
    """
    픽셀 (u, v)를 지면(z=0) 위 실제 좌표로 투영해 거리·방향각을 계산.

    Parameters
    ----------
    u, v : float
        이미지 픽셀 좌표 (u=가로/열, v=세로/행)
    K : (3,3) ndarray
        카메라 내부 파라미터 행렬
    R : (3,3) ndarray
        카메라 자세 회전행렬 (카메라 광학 프레임 → 월드)
    C : (3,) array-like
        카메라의 월드 위치 [Cx, Cy, Cz]  (Cz = 카메라 높이)

    Returns
    -------
    distance : float
        드론 바로 아래점 기준 목표까지의 수평 거리 [m]
    bearing : float
        목표 방향각 [rad] (월드 x축 기준, atan2)
    P : (3,) ndarray
        지면 위 목표의 실제 좌표 [X, Y, 0]
    """
    K = np.asarray(K, dtype=float)
    R = np.asarray(R, dtype=float)
    C = np.asarray(C, dtype=float)

    pixel   = np.array([u, v, 1.0])          # 동차좌표 (2D 점 + 1)
    d_cam   = np.linalg.inv(K) @ pixel       # ① 픽셀 → 카메라 광선 방향
    d_world = R @ d_cam                       # ② → 월드 좌표계 방향

    # ③ 지면(z=0) 교차: 광선이 위로 향하면(d_world_z >= 0) 땅에 안 닿음
    if d_world[2] >= 0:
        raise ValueError(
            "광선이 지면을 향하지 않습니다 (d_world_z >= 0). "
            "카메라 자세 R 또는 광학 프레임 축 방향을 확인하세요."
        )
    t = -C[2] / d_world[2]                    # 지면까지의 비율 (양수여야 정상)
    P = C + t * d_world                        # ④ 땅 위 실제 좌표

    dx, dy   = P[0] - C[0], P[1] - C[1]
    distance = np.sqrt(dx**2 + dy**2)          # 수평 거리
    bearing  = np.arctan2(dy, dx)              # 방향각 (라디안)
    return distance, bearing, P


# ─────────────────────────────────────────────────────────────
# 편의 함수: 똑바로 아래를 보는 카메라의 회전행렬
# (검증 단계용. 실제 시스템에서는 이 R을 TF에서 받아온다.)
# ─────────────────────────────────────────────────────────────
def R_camera_down():
    """카메라가 월드 -z(똑바로 아래)를 바라볼 때의 회전행렬."""
    return np.array([[1,  0,  0],
                     [0, -1,  0],
                     [0,  0, -1]], dtype=float)


# Week 1에서 뽑은 우리 시뮬레이터 카메라의 K
K_SIM = np.array([[554.38,   0.0, 320.5],
                  [  0.0, 554.38, 240.5],
                  [  0.0,   0.0,   1.0]])


# ─────────────────────────────────────────────────────────────
# 단위 테스트 (python3 pixel_to_ground.py 로 실행)
# ─────────────────────────────────────────────────────────────
def _run_unit_tests():
    K = K_SIM
    R = R_camera_down()
    C = np.array([0.0, 0.0, 5.0])   # 원점 위 5m에 뜬 카메라
    TOL = 1e-6

    # 테스트 1: 정중앙 픽셀(주점) → 드론 바로 아래 → 거리 0
    dist, bearing, P = pixel_to_ground(320.5, 240.5, K, R, C)
    assert abs(dist - 0.0) < 1e-4, f"[T1] 거리 {dist} 는 0 이어야 함"
    assert abs(P[2] - 0.0) < TOL,  f"[T1] P_z {P[2]} 는 0 이어야 함"
    print(f"[T1] 정중앙 픽셀      → 거리 {dist:.4f} m  (기대 0)          ✓")

    # 테스트 2: 지면 교차점의 z는 항상 0 이어야 함 (자기일관성)
    for (u, v) in [(100, 100), (500, 300), (639, 479), (0, 0)]:
        _, _, P = pixel_to_ground(u, v, K, R, C)
        assert abs(P[2]) < TOL, f"[T2] ({u},{v}) 에서 P_z {P[2]} != 0"
    print(f"[T2] 임의 픽셀들의 P_z = 0 (지면 위)                         ✓")

    # 테스트 3: 오른쪽으로 120px → +x 방향, 거리 ≈ 1.08 m
    dist, bearing, P = pixel_to_ground(440.5, 240.5, K, R, C)
    exp = 120.0 / 554.38 * 5.0        # (Δu/fx)*높이
    assert abs(dist - exp) < 1e-3,       f"[T3] 거리 {dist} != {exp}"
    assert abs(np.degrees(bearing)) < 0.1, f"[T3] 방향각 {np.degrees(bearing)}° ≈ 0 이어야"
    print(f"[T3] 오른쪽 120px 픽셀 → 거리 {dist:.4f} m, 방향 {np.degrees(bearing):+.1f}°  (기대 {exp:.4f} m, 0°) ✓")

    # 테스트 4: 대칭성 — 좌/우 대칭 픽셀은 거리 같고 방향 180° 차이
    d_r, b_r, _ = pixel_to_ground(440.5, 240.5, K, R, C)  # 오른쪽
    d_l, b_l, _ = pixel_to_ground(200.5, 240.5, K, R, C)  # 왼쪽(대칭)
    assert abs(d_r - d_l) < 1e-6, f"[T4] 좌우 거리 불일치 {d_r} vs {d_l}"
    assert abs(abs(np.degrees(b_r - b_l)) - 180.0) < 1e-6, "[T4] 방향각 180° 차이 아님"
    print(f"[T4] 좌우 대칭 픽셀: 거리 동일({d_r:.4f} m), 방향 180° 차이       ✓")

    # 테스트 5: 위로 향하는 광선(하늘) → 에러 발생해야 정상
    R_up = np.array([[1, 0, 0], [0, -1, 0], [0, 0, 1]], dtype=float)  # z를 위로
    try:
        pixel_to_ground(320.5, 240.5, K, R_up, C)
        raise AssertionError("[T5] 하늘 방향인데 에러가 안 남")
    except ValueError:
        print(f"[T5] 하늘 향하는 광선 → ValueError 로 거부                  ✓")

    print("\n모든 단위 테스트 통과 ✅")


if __name__ == "__main__":
    _run_unit_tests()

    # 데모 출력: 여러 픽셀에 대한 거리·방향각
    print("\n── 데모: 픽셀별 거리/방향각 (카메라 원점 위 5m, 똑바로 아래) ──")
    K, R, C = K_SIM, R_camera_down(), np.array([0.0, 0.0, 5.0])
    for (u, v) in [(320, 240), (440, 240), (320, 120), (500, 400)]:
        dist, bearing, P = pixel_to_ground(u, v, K, R, C)
        print(f"  픽셀({u:3d},{v:3d}) → 거리 {dist:5.3f} m, "
              f"방향 {np.degrees(bearing):+6.1f}°, "
              f"지면좌표 P=({P[0]:+.3f}, {P[1]:+.3f}, {P[2]:.3f})")
