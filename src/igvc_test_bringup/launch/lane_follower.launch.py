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
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import (
    LaunchConfiguration,
    PathJoinSubstitution,
    PythonExpression,
)
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

    force_identity_map_to_odom_arg = DeclareLaunchArgument(
        'force_identity_map_to_odom',
        default_value='true',
        choices=['true', 'false'],
        description=(
            'If true, pin map→odom to identity (0,0,0,0,0,0) for the full run. '
            'This overrides GPS/localization ownership of the transform.'))

    navigator_profile_arg = DeclareLaunchArgument(
        'navigator_profile',
        default_value='autonav',
        choices=['autonav', 'fsd'],
        description='Select navigation behavior profile/executable.')

    use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time',
        default_value='false',
        choices=['true', 'false'],
        description='Use /clock from simulator (true in Gazebo / Isaac Sim).')

    gps_enabled  = LaunchConfiguration('gps_enabled')
    force_identity_map_to_odom = LaunchConfiguration('force_identity_map_to_odom')
    use_sim_time = LaunchConfiguration('use_sim_time')
    navigator_profile = LaunchConfiguration('navigator_profile')

    bringup = FindPackageShare('igvc_test_bringup')

    # ── Shared parameter dict passed to every custom node ─────────────────
    # Both nodes read gps_enabled and use_sim_time so the flag only needs to
    # be set once at launch time.
    shared_params = {
        'use_sim_time': use_sim_time,
        'gps_enabled':  gps_enabled,
        'force_identity_map_to_odom': force_identity_map_to_odom,
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

    # lane_segmentation_node = IncludeLaunchDescription(
    #     PythonLaunchDescriptionSource(
    #         PathJoinSubstitution([bringup, 'launch', 'lane_segmentation.launch.py'])
    #     ),
    #     launch_arguments={
    #         'use_sim_time': use_sim_time,
    #     }.items(),
    # )

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
    navigator_autonav_node = Node(
        package='igvc_lane_detection',
        executable='navigation_autonav_node',
        name='igvc_navigator',
        output='screen',
        parameters=[shared_params],
        condition=IfCondition(PythonExpression([
            "'", navigator_profile, "' == 'autonav'"
        ])),
    )

    navigator_fsd_node = Node(
        package='igvc_lane_detection',
        executable='navigation_fsd_node',
        name='igvc_navigator',
        output='screen',
        parameters=[shared_params],
        condition=IfCondition(PythonExpression([
            "'", navigator_profile, "' == 'fsd'"
        ])),
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
        force_identity_map_to_odom_arg,
        navigator_profile_arg,
        use_sim_time_arg,
        lane_detection_node,
        # lane_segmentation_node,
        localization_node,
        # odom_tf_bridge_node,
        navigator_autonav_node,
        navigator_fsd_node,
        nav2_launch,
    ])

