"""
spartina_moving.launch.py — [Phase 1] 드론 이동 버전 런치
=========================================================
기반: spartina_nav2.launch.py (기존 파일은 손대지 않음)

바뀐 점 딱 2가지:
  1. world : spartina_world.world → spartina_world_moving.world
             (드론 모델 kinematic 화 + gazebo_ros_state 플러그인 추가)
  2. TF    : map→drone_camera_optical 을 발행하던
             static_transform_publisher 를 제거하고,
             drone_tf_broadcaster.py (동적 TF) 로 교체

그대로 유지:
  - map→odom static TF (identity)
  - TurtleBot3 스폰 (-2, 0)
  - Gazebo 서버/클라이언트, robot_state_publisher

실행:
  export TURTLEBOT3_MODEL=waffle
  ros2 launch ~/drone_project/launch/spartina_moving.launch.py

이후 별도 터미널:
  T2  Nav2
  T3  goal_relay_node.py
  T4  obstacle_locator_node.py (world_frame:=map)
  T5  drone_survey_node.py     ← Phase 2 부터
  T6  click_pixel_node.py 또는 탐지기

담당: 정우열 · 스마트해운물류 x ICT 멘토링 (갯끈풀 탐지, 2번 파트)
"""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, TimerAction, ExecuteProcess
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    launch_file_dir = os.path.join(get_package_share_directory('turtlebot3_gazebo'), 'launch')
    pkg_gazebo_ros = get_package_share_directory('gazebo_ros')

    use_sim_time = LaunchConfiguration('use_sim_time', default='true')
    x_pose = LaunchConfiguration('x_pose', default='-2.0')
    y_pose = LaunchConfiguration('y_pose', default='0.0')

    home = os.path.expanduser('~')
    project_dir = os.path.join(home, 'drone_project')
    # ★ 변경점 1: 이동 버전 world
    world = os.path.join(project_dir, 'worlds', 'spartina_world_moving.world')

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
        launch_arguments={'x_pose': x_pose, 'y_pose': y_pose}.items()
    )
    delayed_spawn = TimerAction(period=5.0, actions=[spawn_turtlebot_cmd])

    # ── [TF 1] map → drone_camera_optical ──
    # ★ 변경점 2: static_transform_publisher 를 제거하고 동적 브로드캐스터로 교체.
    #    Gazebo 가 완전히 뜬 뒤 실행해야 /gazebo/model_states 를 놓치지 않으므로 지연 실행.
    drone_tf_cmd = TimerAction(period=7.0, actions=[
        ExecuteProcess(
            cmd=['python3',
                 os.path.join(project_dir, 'scripts', 'drone_tf_broadcaster.py'),
                 '--ros-args', '-p', 'use_sim_time:=true'],
            output='screen',
        )
    ])

    # ── [TF 2] map → odom (identity) ── 기존과 동일
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
    ld.add_action(drone_tf_cmd)
    ld.add_action(static_tf_odom)
    return ld
