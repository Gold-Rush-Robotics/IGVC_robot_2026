"""
igvc_bringup.launch.py

Single launch file for both hardware and simulation.

Hardware:
    ros2 launch igvc_test_bringup igvc_bringup.launch.py

Isaac Sim / GPS-denied:
    ros2 launch igvc_test_bringup igvc_bringup.launch.py \
        gps_enabled:=false use_sim_time:=true

Arguments
    gps_enabled     true | false    default: true
    use_sim_time    true | false    default: false
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import UnlessCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description() -> LaunchDescription:

    # ── Arguments ─────────────────────────────────────────────────────────
    gps_enabled_arg = DeclareLaunchArgument(
        'gps_enabled',
        default_value='false',
        choices=['true', 'false'],
        description=(
            'true  = hardware; monitors GPS health, hands TF to robot_localization.\n'
            'false = sim/GPS-denied; seeds identity map→odom, no GPS checks.'))

    use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time',
        default_value='true',
        choices=['true', 'false'],
        description='Use /clock from simulator (true in Gazebo / Isaac Sim).')

    gps_enabled  = LaunchConfiguration('gps_enabled')
    use_sim_time = LaunchConfiguration('use_sim_time')

    bringup = FindPackageShare('igvc_test_bringup')

    # ── Shared parameter dict passed to every custom node ─────────────────
    # Both nodes read gps_enabled and use_sim_time so the flag only needs to
    # be set once at launch time.
    shared_params = {
        'use_sim_time': use_sim_time,
        'gps_enabled':  gps_enabled,
    }

    # ── Lane detection ────────────────────────────────────────────────────
    lane_detection_node = Node(
        package='igvc_lane_detection',
        executable='lane_detection_node',
        name='lane_detection_node',
        output='screen',
        additional_env={'PYTHONNOUSERSITE': '1'},
        parameters=[
            {'use_sim_time': use_sim_time},
            PathJoinSubstitution([bringup, 'config', 'lane_detection_config.yaml']),
        ],
    )

    # ── Localization (replaces gps_fallback_node) ─────────────────────────
    localization_node = Node(
        package='igvc_lane_detection',
        executable='localization_node',
        name='igvc_localization',
        output='screen',
        parameters=[shared_params],
    )

    odom_tf_bridge_node = Node(
        package='igvc_lane_detection',
        executable='odom_tf_bridge_node',
        name='odom_tf_bridge',
        output='screen',
        parameters=[
            {'use_sim_time': use_sim_time},
            {'odom_topic': '/odom'},
            {'odom_frame_id': 'odom'},
            {'base_frame_id': 'base_link'},
        ],
    )

    # ── Navigator (replaces waypoint_navigator + local_progress_node) ─────
    navigator_node = Node(
        package='igvc_lane_detection',
        executable='navigation_node',
        name='igvc_navigator',
        output='screen',
        parameters=[shared_params],
    )

    # ── Static map→odom TF — sim only ─────────────────────────────────────
    # Provides the TF immediately at startup before igvc_localization's
    # first broadcast tick.  Suppressed on hardware where robot_localization
    # owns the transform.
    static_map_odom_tf = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='map_to_odom_tf',
        output='screen',
        arguments=['0', '0', '0', '0', '0', '0', 'map', 'odom'],
        condition=UnlessCondition(gps_enabled),
    )

    # ── Nav2 ──────────────────────────────────────────────────────────────
    nav2_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([bringup, 'launch', 'navigation_no_docking.launch.py']),
        ]),
        launch_arguments={
            'params_file': PathJoinSubstitution(
                [bringup, 'config', 'nav2_lane_follow_config.yaml']),
            'use_sim_time': use_sim_time,
        }.items(),
    )

    return LaunchDescription([
        gps_enabled_arg,
        use_sim_time_arg,
        lane_detection_node,
        localization_node,
        odom_tf_bridge_node,
        navigator_node,
        static_map_odom_tf,
        nav2_launch,
    ])

