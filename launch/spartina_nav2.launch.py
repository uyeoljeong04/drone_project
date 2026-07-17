"""
spartina_nav2.launch.py — 파트2·3 통합 런치 (Nav2 연동용, map 기준 통일)
=======================================================================
기반: spartina_integration.launch.py (파트2 통합 런치)
바뀐 점 — 좌표계를 전부 map 으로 통일:
  - 기존 world→odom, world→drone_camera_optical (world 기준) 을
    map→odom, map→drone_camera_optical (map 기준) 로 바꿈.
  - 이유: 수환이 Nav2 는 map 프레임 기준으로 도는데(nav2_explore.yaml
    global_frame: map), 예전엔 파트2가 world 를 써서 둘이 안 맞았음.
    작은 시뮬이라 map=odom=Gazebo원점 을 전부 겹치게(identity) 두면
    world 프레임 자체가 필요 없어짐.

이 런치 하나로: Gazebo(월드+카메라) + TurtleBot3 + map→odom + map→카메라 TF.
→ 이후 별도 터미널에서 Nav2 / obstacle_locator / goal_relay 실행.

배치: ~/drone_project/launch/ 에 복사
실행: export TURTLEBOT3_MODEL=waffle
      ros2 launch ~/drone_project/launch/spartina_nav2.launch.py

주의:
  - obstacle_locator 는 반드시 world_frame:=map 로 실행할 것
    (아래 TF 를 map 기준으로 발행하므로)
  - map→odom 을 이 런치가 발행하므로, 수환이 방식의 "터미널2 static TF"
    (map odom) 는 따로 켜지 않는다. (이중 발행 = odom 부모 충돌)
"""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    launch_file_dir = os.path.join(get_package_share_directory('turtlebot3_gazebo'), 'launch')
    pkg_gazebo_ros = get_package_share_directory('gazebo_ros')

    use_sim_time = LaunchConfiguration('use_sim_time', default='true')
    # 로봇 스폰 위치 (Gazebo 좌표). map→odom 이 identity 이므로 map 좌표와 동일.
    x_pose = LaunchConfiguration('x_pose', default='-2.0')
    y_pose = LaunchConfiguration('y_pose', default='0.0')

    home = os.path.expanduser('~')
    world = os.path.join(home, 'drone_project', 'worlds', 'spartina_world.world')

    gzserver_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_gazebo_ros, 'launch', 'gzserver.launch.py')
        ),
        launch_arguments={'world': world}.items()
    )

    gzclient_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_gazebo_ros, 'launch', 'gzclient.launch.py')
        )
    )

    robot_state_publisher_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(launch_file_dir, 'robot_state_publisher.launch.py')
        ),
        launch_arguments={'use_sim_time': use_sim_time}.items()
    )

    spawn_turtlebot_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(launch_file_dir, 'spawn_turtlebot3.launch.py')
        ),
        launch_arguments={
            'x_pose': x_pose,
            'y_pose': y_pose
        }.items()
    )

    delayed_spawn = TimerAction(period=5.0, actions=[spawn_turtlebot_cmd])

    # ── [TF 1] map → drone_camera_optical (카메라 광학 프레임) ──
    # 위치 (0,0,5) + 회전(수직하방 광학 프레임). 파트2 R_HARDCODED 와 동일한 쿼터니언.
    static_tf_camera = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='static_tf_map_to_camera',
        arguments=[
            '--x', '0.0', '--y', '0.0', '--z', '5.0',
            '--qx', '0.7071068', '--qy', '-0.7071068',
            '--qz', '0.0', '--qw', '0.0',
            '--frame-id', 'map',
            '--child-frame-id', 'drone_camera_optical',
        ],
    )

    # ── [TF 2] map → odom (identity) ── 수환이 "터미널2" 역할을 여기서 대체.
    # TurtleBot3 diff_drive 가 odom 을 Gazebo 원점 기준으로 발행하므로 identity.
    static_tf_odom = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='static_tf_map_to_odom',
        arguments=[
            '--x', '0.0', '--y', '0.0', '--z', '0.0',
            '--yaw', '0.0', '--pitch', '0.0', '--roll', '0.0',
            '--frame-id', 'map',
            '--child-frame-id', 'odom',
        ],
    )

    ld = LaunchDescription()
    ld.add_action(gzserver_cmd)
    ld.add_action(gzclient_cmd)
    ld.add_action(robot_state_publisher_cmd)
    ld.add_action(delayed_spawn)
    ld.add_action(static_tf_camera)
    ld.add_action(static_tf_odom)
    return ld
