"""
spartina_integration.launch.py — 파트2·3 통합 런치
====================================================
기반: 여수환 spartina_sim.launch.py
추가된 것 (2가지 static TF):
  1. world → drone_camera_optical : 카메라 pose(0,0,5, 수직하방)의 광학 프레임.
     쿼터니언 (0.7071068, -0.7071068, 0, 0) 은 2번 파트 검증에 쓰인
     R_HARDCODED = [[0,-1,0],[-1,0,0],[0,0,-1]] 와 정확히 동일한 회전.
  2. world → odom : TurtleBot3 TF 트리(odom→base_footprint→base_link)의
     뿌리인 odom 을 world 에 연결. odom 원점 = 로봇 스폰 위치이므로
     스폰 좌표(x_pose, y_pose)와 반드시 일치해야 함.

이 런치 하나로: Gazebo(월드+카메라) + TurtleBot3 + TF 완성.
이후 별도 터미널에서 obstacle_locator_node / target_follower_node 실행.

배치: ~/drone_project/launch/ 에 복사
실행: ros2 launch ./spartina_integration.launch.py
      (또는 기존 방식대로 패키지에 넣고 실행)
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
    # 로봇 스폰 위치 — 아래 world→odom TF 와 반드시 같은 값이어야 함
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

    # ── [신규 1] world → drone_camera_optical (카메라 광학 프레임) ──
    # 위치 (0,0,5) + 회전 R_HARDCODED (수직하방 광학 프레임)
    static_tf_camera = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='static_tf_world_to_camera',
        arguments=[
            '--x', '0.0', '--y', '0.0', '--z', '5.0',
            '--qx', '0.7071068', '--qy', '-0.7071068',
            '--qz', '0.0', '--qw', '0.0',
            '--frame-id', 'world',
            '--child-frame-id', 'drone_camera_optical',
        ],
    )

    # ── [신규 2] world → odom (로봇 TF 트리를 world 에 연결) ──
    # ※ TurtleBot3 diff_drive 플러그인은 odom 을 Gazebo 원점 기준(world 소스)으로
    #   발행하므로 world→odom 은 항등(0,0)이어야 함. 스폰 오프셋을 넣으면
    #   이중 적용되어 로봇 위치가 실제보다 2배 밀림 (실측으로 확인).
    static_tf_odom = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='static_tf_world_to_odom',
        arguments=[
            '--x', '0.0', '--y', '0.0', '--z', '0.0',
            '--yaw', '0.0', '--pitch', '0.0', '--roll', '0.0',
            '--frame-id', 'world',
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
