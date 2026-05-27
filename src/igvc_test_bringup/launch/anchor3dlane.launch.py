"""
anchor3dlane.launch.py

Runs the Anchor3DLane++ 3D lane detection node alongside the IGVC navigation
stack.  Drop-in alternative to ``lane_segmentation.launch.py`` — it swaps
YOLOPv2 for Anchor3DLane++ while keeping the same navigator / mission planner
pipeline.  Both nodes publish on identical topics (/lane_costmap, /lane_map,
/lane_segmentation/lanes) so no downstream changes are required.

Prerequisites
-------------
See ``anchor3dlane_infer.py`` for the one-time install of mmcv / Anchor3DLane.

Typical usage
-------------
    # Set paths in the config file first, then:
    ros2 launch igvc_test_bringup anchor3dlane.launch.py

    # Override model paths at launch time:
    ros2 launch igvc_test_bringup anchor3dlane.launch.py \\
        config_path:=/home/user/anchor3dlane/configs/Anchor3DLane/anchor3dlane_plusplus_r18_openlane.py \\
        checkpoint_path:=/home/user/weights/anchor3dlane_plusplus_r18_360x480.pth

    # Simulation (GPS-denied):
    ros2 launch igvc_test_bringup anchor3dlane.launch.py \\
        use_sim_time:=true gps_enabled:=false

Arguments
---------
    gps_enabled                  true | false       default: false
    force_identity_map_to_odom   true | false       default: true
    use_sim_time                 true | false       default: false
    anchor3dlane_root            path               default: "" (already installed)
    config_path                  path               default: ""
    checkpoint_path              path               default: ""
    model_device                 cpu | cuda:N       default: cuda:0
    model_half                   true | false       default: true
    score_threshold              float              default: 0.4
    publish_overlay              true | false       default: true
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def _make_nodes(context, *args, **kwargs):
    """Build node list at launch-time so empty string args don't override YAML."""
    def _get(name: str) -> str:
        return LaunchConfiguration(name).perform(context)

    bringup = FindPackageShare('igvc_test_bringup').perform(context)
    yaml    = bringup + '/config/anchor3dlane_config.yaml'

    # Only override model-path params when the launch arg was actually provided
    # (non-empty). Otherwise the YAML values are used as-is.
    overrides: dict = {
        'use_sim_time':    _get('use_sim_time') == 'true',
        'model_device':    _get('model_device'),
        'model_half':      _get('model_half') == 'true',
        'score_threshold': float(_get('score_threshold')),
        'publish_overlay': _get('publish_overlay') == 'true',
    }
    for key in ('anchor3dlane_root', 'config_path', 'checkpoint_path'):
        val = _get(key)
        if val:  # only add when non-empty so YAML default is kept
            overrides[key] = val

    use_sim_time_bool = _get('use_sim_time') == 'true'
    force_identity    = _get('force_identity_map_to_odom') == 'true'
    gps_enabled       = _get('gps_enabled') == 'true'

    shared_params = {
        'use_sim_time': use_sim_time_bool,
        'gps_enabled':  gps_enabled,
        'force_identity_map_to_odom': force_identity,
        'max_odom_age_sec': 0.75,
    }

    anchor3dlane_node = Node(
        package='igvc_lane_detection',
        executable='anchor3dlane_node',
        name='anchor3dlane_node',
        output='screen',
        parameters=[yaml, overrides],
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
        parameters=[{'use_sim_time': use_sim_time_bool}],
        condition=IfCondition(str(force_identity).lower()),
    )

    mission_planner_node = Node(
        package='igvc_lane_detection',
        executable='mission_planner_node',
        name='mission_planner_node',
        output='screen',
        parameters=[
            shared_params,
            bringup + '/config/mission_planner_config.yaml',
        ],
    )

    navigator_node = Node(
        package='igvc_lane_detection',
        executable='navigation_node',
        name='navigation_node',
        output='screen',
        parameters=[
            shared_params,
            bringup + '/config/navigator_config.yaml',
        ],
    )

    odom_tf_bridge_node = Node(
        package='igvc_lane_detection',
        executable='odom_tf_bridge_node',
        name='odom_tf_bridge_node',
        output='screen',
        parameters=[
            shared_params,
            bringup + '/config/odom_tf_bridge_config.yaml',
        ],
    )

    return [
        anchor3dlane_node,
        static_map_to_odom,
        mission_planner_node,
        navigator_node,
        odom_tf_bridge_node,
    ]


def generate_launch_description() -> LaunchDescription:

    # ── Arguments ─────────────────────────────────────────────────────────

    gps_enabled_arg = DeclareLaunchArgument(
        'gps_enabled', default_value='false', choices=['true', 'false'])
    force_identity_arg = DeclareLaunchArgument(
        'force_identity_map_to_odom', default_value='true',
        choices=['true', 'false'])
    use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time', default_value='false', choices=['true', 'false'])

    anchor3dlane_root_arg = DeclareLaunchArgument(
        'anchor3dlane_root', default_value='',
        description='Root of the cloned Anchor3DLane repo; added to sys.path.')
    config_path_arg = DeclareLaunchArgument(
        'config_path', default_value='',
        description='Absolute path to the mmcv config .py file.')
    checkpoint_path_arg = DeclareLaunchArgument(
        'checkpoint_path', default_value='',
        description='Absolute path to the .pth checkpoint file.')
    model_device_arg = DeclareLaunchArgument(
        'model_device', default_value='cuda:0',
        description='Torch device string (cpu, cuda:0, …).')
    model_half_arg = DeclareLaunchArgument(
        'model_half', default_value='true', choices=['true', 'false'])
    score_threshold_arg = DeclareLaunchArgument(
        'score_threshold', default_value='0.4',
        description='Minimum lane confidence (0–1). Lower = more recalls, more FP.')
    publish_overlay_arg = DeclareLaunchArgument(
        'publish_overlay', default_value='true', choices=['true', 'false'])

    return LaunchDescription([
        # Args
        gps_enabled_arg,
        force_identity_arg,
        use_sim_time_arg,
        anchor3dlane_root_arg,
        config_path_arg,
        checkpoint_path_arg,
        model_device_arg,
        model_half_arg,
        score_threshold_arg,
        publish_overlay_arg,
        # Nodes built at launch-time so empty args don't override YAML values
        OpaqueFunction(function=_make_nodes),
    ])
