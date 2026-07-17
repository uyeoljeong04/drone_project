# drone_project — 파트2·3 통합 (갯끈풀 탐지·자율주행)

노트북 ↔ 학교 우분투 PC 사이 코드 동기화용 저장소.

## 폴더 구성
```
launch/   spartina_nav2.launch.py         통합 런치 (Gazebo+차+카메라+map기준 TF)
          spartina_integration.launch.py  이전 world기준 런치 (참고용)
worlds/   spartina_world.world            카메라 포함 갯벌 월드 (목표 2,0)
config/   nav2_explore.yaml               Nav2 설정 (탐험모드)
          cyclonedds.xml                  DDS 설정
scripts/  obstacle_locator_node.py        [파트2] 픽셀→world좌표 계산
          goal_relay_node.py              [통합] world좌표→/goal_pose 변환 다리
          click_pixel_node.py             [파트2] 카메라 영상 클릭 → 픽셀
          target_follower_node.py         [파트2] 단순 추종 (Nav2 쓰면 미사용)
          pixel_to_ground.py, validate_projection.py  [파트2] 좌표 수식/검증
```

## 학교 우분투에 처음 세팅 (한 번만)
```bash
sudo apt update
sudo apt install ros-humble-navigation2 ros-humble-nav2-bringup -y
# 이 저장소를 clone 후, 필요하면 심볼릭 링크나 복사로 ~/drone_project 로 사용
```

## 실행 순서 (터미널 여러 개)
```bash
# T1  시뮬 + 차 + TF(map→odom, map→카메라)
export TURTLEBOT3_MODEL=waffle
ros2 launch ~/drone_project/launch/spartina_nav2.launch.py

# T2  Nav2 (탐험모드)
export TURTLEBOT3_MODEL=waffle
ros2 launch nav2_bringup navigation_launch.py use_sim_time:=True \
  params_file:=$HOME/drone_project/config/nav2_explore.yaml

# T3  변환 다리
python3 ~/drone_project/scripts/goal_relay_node.py

# T4  파트2 좌표계산 (world_frame=map 필수!)
python3 ~/drone_project/scripts/obstacle_locator_node.py --ros-args \
  -p use_tf2:=true -p world_frame:=map \
  -p camera_frame:=drone_camera_optical -p robot_frame:=base_link

# T5  목표 입력 — 고정 픽셀 or 클릭
ros2 topic pub --once /detection/pixel geometry_msgs/msg/Point "{x: 320.0, y: 240.0}"
#   또는
python3 ~/drone_project/scripts/click_pixel_node.py
```

주의: map→odom 을 런치가 발행하므로 수환이 방식의 "터미널2 static TF"는 켜지 않는다.
