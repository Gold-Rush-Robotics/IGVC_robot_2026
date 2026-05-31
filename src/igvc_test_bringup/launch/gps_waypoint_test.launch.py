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
    params_file         Nav2 params YAML                    (default: nav2_gps_waypoint_config.yaml)
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
        default_value=PathJoinSubstitution([bringup, 'config', 'nav2_gps_waypoint_config.yaml']),
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
    # This node also owns the map -> odom transform: it publishes identity
    # until it has driven forward to calibrate true heading, then republishes
    # the corrected transform.  Do NOT also run a static map->odom publisher
    # here — two publishers on the same parent/child fight in the TF tree.
    gps_waypoint_test_node = Node(
        package='igvc_lane_detection',
        executable='gps_waypoint_test_node',
        name='gps_waypoint_test',
        output='screen',
        parameters=[{
            'use_sim_time': use_sim_time,
            # ── Mode ────────────────────────────────────────────────────────
            # use_gps=false: navigate to a fixed map-frame pose using odometry
            # only.  The robot starts at (0, 0) and drives to (target_x, target_y).
            'use_gps': True,
            'target_x': 0.0,   # metres forward (+X)
            'target_y': 0.0,   # metres lateral (+Y = left)
            # ── GPS target (ignored when use_gps=false) ─────────────────────
            'target_lat': 42.66821182,
            'target_lon': -83.21845873,
            'origin_lat': origin_lat,
            'origin_lon': origin_lon,
            'odom_topic': '/front_zed_camera_x/zed_node/odom',
            'map_frame': 'map',
            'odom_frame': 'odom',
            # ── Heading triangulation (3 GPS fixes) ─────────────────────────
            # P1: average GPS at start.  Drive 1 m → P2.  Drive 1 m → P3.
            # Heading = P1→P3 vector; if atan2 returns a flipped result the
            # node auto-corrects using the odom displacement as a reference.
            'heading_init': True,
            'calib_distance_m': 1.0,
            'calib_speed_mps': 0.3,
            'calib_settle_sec': 2.0,
            'drive_cmd_topic': 'cmd_vel_nav',
            'min_gps_displacement_m': 0.5,
            'recovery_dist_increase_m': 2.0,
            'max_recoveries': 5,
            # ── Closed-loop GPS regression ───────────────────────────────────
            # Rolling window of GPS fixes fed into the linear regression that
            # smooths the current position estimate during navigation.
            'gps_regression_window_sec': 5.0,
            'gps_min_samples': 3,
            # Resend the Nav2 goal whenever the regressed GPS position moves
            # this far from where the last goal was sent.  Keep active-goal
            # refresh disabled so Nav2 can finish the current route instead of
            # repeatedly accepting preemptive NavigateThroughPoses requests.
            'goal_update_distance_m': 0.5,
            'allow_active_goal_refresh': False,
        }],
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
    twist_stamper_node = Node(
        package='twist_stamper',
        executable='twist_stamper',
        name='twist_stamper',
        output='screen',
        parameters=[
            {'use_sim_time': use_sim_time},
            {'frame_id': 'base_link'},
        ],
        remappings=[
            ('cmd_vel_in', '/diff_drive_controller/cmd_vel_unstamped'),
            ('cmd_vel_out', '/diff_drive_controller/cmd_vel'),
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
        odom_tf_bridge_node,
        twist_stamper_node,
    ])
