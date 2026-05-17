"""
lane_segmentation.launch.py

Runs the YOLOPv2 segmentation-based lane node alongside the usual IGVC
stack (localization, navigator, Nav2).  This is a drop-in alternative to
``lane_follower.launch.py`` — it keeps the same navigation pipeline but
swaps the Hough lane detector for the deep segmentation model.

Typical usage:

    # Hardware (weights exported in the shell)
    export YOLOPV2_WEIGHTS=$PWD/models/yolopv2.pt
    ros2 launch igvc_test_bringup lane_segmentation.launch.py

    # Isaac Sim / GPS-denied
    ros2 launch igvc_test_bringup lane_segmentation.launch.py \
        gps_enabled:=false use_sim_time:=true \
        model_weights:=/abs/path/to/yolopv2.pt

Arguments
    gps_enabled                  true | false       default: false
    force_identity_map_to_odom   true | false       default: true
    use_sim_time                 true | false       default: false
    model_weights                absolute path      default: $YOLOPV2_WEIGHTS
    model_device                 cpu | cuda:N       default: cuda:0
    model_half                   true | false       default: true
    publish_overlay              true | false       default: true
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import (
    EnvironmentVariable,
    LaunchConfiguration,
    PathJoinSubstitution,
)
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description() -> LaunchDescription:

    # ── Arguments ─────────────────────────────────────────────────────

    gps_enabled_arg = DeclareLaunchArgument(
        'gps_enabled', default_value='false', choices=['true', 'false'])
    force_identity_map_to_odom_arg = DeclareLaunchArgument(
        'force_identity_map_to_odom', default_value='true',
        choices=['true', 'false'])
    use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time', default_value='false', choices=['true', 'false'])

    model_weights_arg = DeclareLaunchArgument(
        'model_weights',
        default_value=EnvironmentVariable('YOLOPV2_WEIGHTS', default_value=''),
        description=(
            'Absolute path to yolopv2.pt TorchScript weights. Falls back to '
            '$YOLOPV2_WEIGHTS. Fetch with '
            'src/igvc_lane_detection/scripts/fetch_yolopv2_weights.sh.'))
    model_device_arg = DeclareLaunchArgument(
        'model_device', default_value='cuda:0',
        description='Torch device string (cpu, cuda:0, …).')
    model_half_arg = DeclareLaunchArgument(
        'model_half', default_value='true', choices=['true', 'false'],
        description='Run the network in FP16 on CUDA devices.')
    publish_overlay_arg = DeclareLaunchArgument(
        'publish_overlay', default_value='true', choices=['true', 'false'])

    gps_enabled  = LaunchConfiguration('gps_enabled')
    force_identity_map_to_odom = LaunchConfiguration('force_identity_map_to_odom')
    use_sim_time = LaunchConfiguration('use_sim_time')
    model_weights = LaunchConfiguration('model_weights')
    model_device  = LaunchConfiguration('model_device')
    model_half    = LaunchConfiguration('model_half')
    publish_overlay = LaunchConfiguration('publish_overlay')

    bringup = FindPackageShare('igvc_test_bringup')

    shared_params = {
        'use_sim_time': use_sim_time,
        'gps_enabled':  gps_enabled,
        'force_identity_map_to_odom': force_identity_map_to_odom,
        'max_odom_age_sec': 0.75,
    }

    # ── YOLOPv2 lane segmentation ─────────────────────────────────────
    lane_segmentation_node = Node(
        package='igvc_lane_detection',
        executable='lane_segmentation_node',
        name='lane_segmentation_node',
        output='screen',
        # NOTE: do NOT set PYTHONNOUSERSITE here — torch is typically
        # installed in the user site-packages (`~/.local/...`) on both
        # the x86 dev machine and the Jetson AGX Orin JetPack image, and
        # hiding user site would break `import torch`.
        parameters=[
            PathJoinSubstitution(
                [bringup, 'config', 'lane_segmentation_config.yaml']),
            {
                'use_sim_time': use_sim_time,
                'model_weights': model_weights,
                'model_device': model_device,
                'model_half': model_half,
                'publish_overlay': publish_overlay,
            },
        ],
    )

    # ── Localization ──────────────────────────────────────────────────
    localization_node = Node(
        package='igvc_lane_detection',
        executable='localization_node',
        name='igvc_localization',
        output='screen',
        parameters=[shared_params],
    )

    # ── Navigator ─────────────────────────────────────────────────────
    navigator_node = Node(
        package='igvc_lane_detection',
        executable='navigation_node',
        name='igvc_navigator',
        output='screen',
        parameters=[
            shared_params,
            PathJoinSubstitution([bringup, 'config', 'navigator_config.yaml']),
        ],
    )

    # ── Nav2 ──────────────────────────────────────────────────────────
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
        use_sim_time_arg,
        model_weights_arg,
        model_device_arg,
        model_half_arg,
        publish_overlay_arg,
        lane_segmentation_node,
        localization_node,
        # static_map_to_odom,
        navigator_node,
        nav2_launch,
    ])
