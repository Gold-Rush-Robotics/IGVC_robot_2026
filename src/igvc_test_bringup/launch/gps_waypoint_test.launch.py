"""
gps_waypoint_test.launch.py

Bring-up for field-testing the GPS → Nav2 localisation chain.

Starts:
  1. Motor controllers  (ros2_control + diff_drive + robot_state_publisher)
  2. Nav2 stack         (navigation_no_docking.launch.py)
  3. gps_waypoint_test_node  — drives the robot to a single GPS coordinate

Usage
-----
    ros2 launch igvc_test_bringup gps_waypoint_test.launch.py \\
        target_lat:=42.678920 target_lon:=-83.195610

Arguments
---------
    target_lat          Target latitude                     (default: 42.400510946)
    target_lon          Target longitude                    (default: -83.130640432)
    origin_lat          Map-origin latitude   (0.0 = first fix)  (default: 0.0)
    origin_lon          Map-origin longitude  (0.0 = first fix)  (default: 0.0)
    hardware_interface  CanInterface | IsaacDriveHardware   (default: CanInterface)
    use_sim_time        true | false                        (default: false)
    params_file         Nav2 params YAML                    (default: nav2_lane_follow_config.yaml)
"""

import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description() -> LaunchDescription:
    bringup = FindPackageShare('igvc_test_bringup')

    # ── Arguments ─────────────────────────────────────────────────────────
    use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time',
        default_value='false',
        description='Use /clock from simulator (true in Gazebo / Isaac Sim).',
    )
    hardware_interface_arg = DeclareLaunchArgument(
        'hardware_interface',
        default_value='CanInterface',
        description='Hardware interface: CanInterface or IsaacDriveHardware.',
    )
    params_file_arg = DeclareLaunchArgument(
        'params_file',
        default_value=PathJoinSubstitution([bringup, 'config', 'nav2_lane_follow_config.yaml']),
        description='Full path to the Nav2 YAML params file.',
    )
    target_lat_arg = DeclareLaunchArgument(
        'target_lat',
        default_value='42.400510946',
        description='Target waypoint latitude.',
    )
    target_lon_arg = DeclareLaunchArgument(
        'target_lon',
        default_value='-83.130640432',
        description='Target waypoint longitude.',
    )
    origin_lat_arg = DeclareLaunchArgument(
        'origin_lat',
        default_value='0.0',
        description='Map-origin latitude. 0.0 anchors to the first GPS fix.',
    )
    origin_lon_arg = DeclareLaunchArgument(
        'origin_lon',
        default_value='0.0',
        description='Map-origin longitude. 0.0 anchors to the first GPS fix.',
    )

    use_sim_time      = LaunchConfiguration('use_sim_time')
    hardware_interface = LaunchConfiguration('hardware_interface')
    params_file       = LaunchConfiguration('params_file')
    target_lat        = LaunchConfiguration('target_lat')
    target_lon        = LaunchConfiguration('target_lon')
    origin_lat        = LaunchConfiguration('origin_lat')
    origin_lon        = LaunchConfiguration('origin_lon')

    # ── 1. Motor controllers ───────────────────────────────────────────────
    motor_controllers = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([bringup, 'launch', 'motor_controllers.launch.py'])
        ),
        launch_arguments={
            'hardware_interface': hardware_interface,
            'use_sim_time': use_sim_time,
        }.items(),
    )

    # ── 2. Nav2 stack ──────────────────────────────────────────────────────
    nav2 = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([bringup, 'launch', 'navigation_no_docking.launch.py'])
        ),
        launch_arguments={
            'use_sim_time': use_sim_time,
            'params_file': params_file,
            'autostart': 'true',
        }.items(),
    )

    # ── 3. GPS waypoint test node ──────────────────────────────────────────
    gps_waypoint_test_node = Node(
        package='igvc_lane_detection',
        executable='gps_waypoint_test_node',
        name='gps_waypoint_test',
        output='screen',
        parameters=[{
            'use_sim_time': use_sim_time,
            'target_lat': 42.66821182,
            'target_lon': -83.21845873,
            'origin_lat': origin_lat,
            'origin_lon': origin_lon,
        }],
    )
    static_map_to_odom = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='static_map_to_odom',
        output='log',
        arguments=[
            '--x', '0', '--y', '0', '--z', '0',
            '--roll', '0', '--pitch', '0', '--yaw', '0',
            '--frame-id', 'map', '--child-frame-id', 'odom',
        ],
        parameters=[{'use_sim_time': use_sim_time}],
    )
    odom_tf_bridge_node = Node(
        package="igvc_lane_detection",
        executable="odom_tf_bridge_node",
        name="odom_tf_bridge",
        output="screen",
        parameters=[
            {'use_sim_time': use_sim_time},
            {'odom_topic': '/front_zed_camera_x/zed_node/odom'},
            {'odom_frame_id': 'odom'},
            {'base_frame_id': 'base_link'},
            {'publish_rate_hz': 100.0},
            # Match isaac_nav_test: publish TF at the current ROS time so
            # RViz/Nav2 can transform base_link-framed /lane_costmap even
            # when the ZED odom message timestamp lags under sim/YOLO load.
            {'use_original_timestamp': False},
            {'warn_odom_age_sec': 0.5},
        ],
    )

    return LaunchDescription([
        use_sim_time_arg,
        hardware_interface_arg,
        params_file_arg,
        target_lat_arg,
        target_lon_arg,
        origin_lat_arg,
        origin_lon_arg,
        motor_controllers,
        nav2,
        gps_waypoint_test_node,
        static_map_to_odom,
        odom_tf_bridge_node,
    ])
