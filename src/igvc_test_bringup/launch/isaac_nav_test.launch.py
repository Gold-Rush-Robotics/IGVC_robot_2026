"""
isaac_nav_test.launch.py

Navigator ground-truth test harness for Isaac Sim.

Instead of running the YOLOPv2 vision pipeline, this launch file:
  1. Publishes the perfect lane occupancy grid from the track geometry
     (track_ground_truth_node → /lane_ground_truth)
  2. Merges obstacle lethal cells into the grid and republishes it as both
     /lane_map  (nav2 StaticLayer) and /lane_costmap  (navigator)
     (gt_nav_bridge_node)
  3. Starts the full nav2 stack (controller_server, planner_server,
     bt_navigator …) using nav2_lane_follow_config.yaml — unchanged from
     the real-world config.
  4. Starts the IGVC navigator node in local_lane + FollowPath mode, so
      planning depends only on /lane_map and /lane_costmap (no pre-known
      centerline waypoints).

Usage
-----
    # Local lane extraction from /lane_costmap + FollowPath
    ros2 launch igvc_test_bringup isaac_nav_test.launch.py

    # Override track
    ros2 launch igvc_test_bringup isaac_nav_test.launch.py \\
        track_file:=/abs/path/to/track_points.json \\
        track_image_file:=/abs/path/to/track.png

Arguments
---------
    use_sim_time            true | false                    (default: true)
    track_file              path to track_points.json
    track_image_file        path to track.png
    track_image_pixels_per_meter  float                     (default: 52.5)
    debug_png_path          GT grid PNG export path
    field_usd_path          Isaac Sim field USD
    robot_usd_path          Isaac Sim robot USD
    robot_entity_name       spawned entity name             (default: igvc_robot)
    robot_spawn_{x,y,z}     world frame spawn coords        (default: from JSON)
    robot_spawn_yaw_rad     spawn yaw                       (default: from JSON)
"""

import json
import os
from pathlib import Path

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.launch_description_sources import (
    AnyLaunchDescriptionSource,
    PythonLaunchDescriptionSource,
)
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


# ── Workspace root detection (shared with isaac_lane_test) ────────────────────

def _resolve_workspace_root() -> str:
    candidates = []
    env_root = os.environ.get('IGVC_WORKSPACE_ROOT', '')
    if env_root:
        candidates.append(os.path.abspath(env_root))
    candidates.append(os.getcwd())
    here = os.path.dirname(__file__)
    # Follow symlink: --symlink-install makes __file__ a symlink inside the
    # install tree; os.path.realpath resolves it back to the source tree.
    real_here = os.path.dirname(os.path.realpath(__file__))
    candidates.append(os.path.abspath(os.path.join(real_here, '..', '..', '..')))
    candidates.append(os.path.abspath(os.path.join(here, '..', '..', '..')))
    candidates.append(os.path.abspath(os.path.join(here, '..', '..', '..', '..', '..')))
    for root in candidates:
        if os.path.isdir(os.path.join(root, 'IGVC_track_generator')):
            return root
    return os.path.abspath(os.path.join(here, '..', '..', '..'))


# ── Launch description ────────────────────────────────────────────────────────

def generate_launch_description() -> LaunchDescription:
    bringup = FindPackageShare('igvc_test_bringup')
    workspace_root = _resolve_workspace_root()
    default_track_file = os.path.join(
        workspace_root, 'IGVC_track_generator', 'track_points.json')
    default_track_image = os.path.join(
        workspace_root, 'IGVC_track_generator', 'track.png')

    # ── LaunchConfiguration aliases ──────────────────────────────────────
    use_sim_time = LaunchConfiguration('use_sim_time')
    track_file = LaunchConfiguration('track_file')
    track_image_file = LaunchConfiguration('track_image_file')
    track_image_pixels_per_meter = LaunchConfiguration('track_image_pixels_per_meter')
    debug_png_path = LaunchConfiguration('debug_png_path')
    field_usd_path = LaunchConfiguration('field_usd_path')
    robot_usd_path = LaunchConfiguration('robot_usd_path')
    robot_entity_name = LaunchConfiguration('robot_entity_name')
    robot_spawn_x = LaunchConfiguration('robot_spawn_x')
    robot_spawn_y = LaunchConfiguration('robot_spawn_y')
    robot_spawn_z = LaunchConfiguration('robot_spawn_z')
    robot_spawn_yaw_rad = LaunchConfiguration('robot_spawn_yaw_rad')
    odom_topic = LaunchConfiguration('odom_topic')

    auto_value = '__auto__'
    fallback_x = 8.128949212924178
    fallback_y = 16.63174225312982
    fallback_z = 0.0820257550378943
    fallback_yaw = -3.141592653589793

    # ── Robot pose from JSON (identical to isaac_lane_test) ──────────────
    def apply_robot_pose_from_json(context, *args, **kwargs):
        track_file_str = context.launch_configurations.get('track_file', '')
        use_auto_x = context.launch_configurations.get('robot_spawn_x', auto_value) == auto_value
        use_auto_y = context.launch_configurations.get('robot_spawn_y', auto_value) == auto_value
        use_auto_z = context.launch_configurations.get('robot_spawn_z', auto_value) == auto_value
        use_auto_yaw = context.launch_configurations.get(
            'robot_spawn_yaw_rad', auto_value) == auto_value
        needs_auto = use_auto_x or use_auto_y or use_auto_z or use_auto_yaw

        if not needs_auto:
            return []

        def _set_defaults():
            if use_auto_x:
                context.launch_configurations['robot_spawn_x'] = str(fallback_x)
            if use_auto_y:
                context.launch_configurations['robot_spawn_y'] = str(fallback_y)
            if use_auto_z:
                context.launch_configurations['robot_spawn_z'] = str(fallback_z)
            if use_auto_yaw:
                context.launch_configurations['robot_spawn_yaw_rad'] = str(fallback_yaw)

        if not track_file_str:
            _set_defaults()
            return []
        try:
            track_path = Path(track_file_str).expanduser()
            if not track_path.exists() or track_path.suffix.lower() != '.json':
                _set_defaults()
                return []
            payload = json.loads(track_path.read_text(encoding='utf-8'))
            if 'robot_start_pose' not in payload:
                _set_defaults()
                return []
            pose = payload['robot_start_pose']
            if 'position_m' in pose and 'yaw_rad' in pose:
                yaw = float(pose.get('yaw_rad', fallback_yaw))
                if use_auto_x:
                    x = float(pose['position_m'].get('x', fallback_x))
                    context.launch_configurations['robot_spawn_x'] = str(x)
                if use_auto_y:
                    y = float(pose['position_m'].get('y', fallback_y))
                    context.launch_configurations['robot_spawn_y'] = str(y)
                if use_auto_z:
                    z = float(pose['position_m'].get('z', fallback_z))
                    context.launch_configurations['robot_spawn_z'] = str(z)
                if use_auto_yaw:
                    context.launch_configurations['robot_spawn_yaw_rad'] = str(yaw)
            else:
                _set_defaults()
        except Exception:
            _set_defaults()
        return []

    # ── Sim interface ─────────────────────────────────────────────────────
    sim_interface = IncludeLaunchDescription(
        AnyLaunchDescriptionSource(
            PathJoinSubstitution([bringup, 'launch', 'simulation_interface.launch.yaml'])
        ),
        launch_arguments={
            'use_sim_time': use_sim_time,
            'field_usd_path': field_usd_path,
            'robot_usd_path': robot_usd_path,
            'robot_entity_name': robot_entity_name,
            'robot_spawn_x': robot_spawn_x,
            'robot_spawn_y': robot_spawn_y,
            'robot_spawn_z': robot_spawn_z,
            'robot_spawn_yaw_rad': robot_spawn_yaw_rad,
            'auto_play': 'false',
        }.items(),
    )

    shared_cfg = PathJoinSubstitution([bringup, 'config', 'isaac_nav_test.yaml'])

    # ── Drive hardware bridge ─────────────────────────────────────────────
    # Nav2 outputs Twist commands, ros2_control converts them to wheel joint
    # velocity commands, and IsaacDriveHardware publishes /isaac_joint_cmd.
    motor_controllers = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([bringup, 'launch', 'motor_controllers.launch.py'])
        ),
        launch_arguments={
            'hardware_interface': 'IsaacDriveHardware',
            'use_sim_time': use_sim_time,
        }.items(),
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

    odom_tf_bridge_node = Node(
        package='igvc_lane_detection',
        executable='odom_tf_bridge_node',
        name='odom_tf_bridge',
        output='screen',
        remappings=[('/odom', odom_topic)],
        parameters=[{
            'use_sim_time': use_sim_time,
            'odom_topic': '/odom',
            'odom_frame_id': 'odom',
            'base_frame_id': 'base_link',
            'publish_rate_hz': 100.0,
            'use_original_timestamp': False,
            'warn_odom_age_sec': 0.5,
        }],
    )

    # ── Ground-truth grid ─────────────────────────────────────────────────
    gt_node = Node(
        package='igvc_lane_detection',
        executable='track_ground_truth_node',
        name='track_ground_truth_node',
        output='screen',
        parameters=[
            shared_cfg,
            {
                'use_sim_time': use_sim_time,
                'track_file': track_file,
                'track_image_file': track_image_file,
                'track_image_pixels_per_meter': track_image_pixels_per_meter,
                'debug_png_path': debug_png_path,
            },
        ],
    )

    # ── GT → /lane_map + /lane_costmap bridge with obstacle stamping ──────
    gt_bridge_node = Node(
        package='igvc_lane_detection',
        executable='gt_nav_bridge_node',
        name='gt_nav_bridge_node',
        output='screen',
        remappings=[('/odom', odom_topic)],
        parameters=[
            shared_cfg,
            {
                'use_sim_time': use_sim_time,
                'track_file': track_file,
            },
        ],
    )

    # ── Localization: publishes identity map→odom TF in sim ───────────────
    localization_node = Node(
        package='igvc_lane_detection',
        executable='localization_node',
        name='igvc_localization',
        output='screen',
        remappings=[('/odom', odom_topic)],
        parameters=[{
            'use_sim_time': use_sim_time,
            'gps_enabled': False,
            'force_identity_map_to_odom': True,
            'max_odom_age_sec': 0.5,
        }],
    )

    # ── Nav2 stack ────────────────────────────────────────────────────────
    nav2 = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([bringup, 'launch', 'navigation_no_docking.launch.py'])
        ),
        launch_arguments={
            'params_file': PathJoinSubstitution(
                [bringup, 'config', 'nav2_lane_follow_config.yaml']),
            'extra_params_file': PathJoinSubstitution(
                [bringup, 'config', 'isaac_nav_test_nav2_overrides.yaml']),
            'use_sim_time': use_sim_time,
        }.items(),
    )

    # ── Navigator ─────────────────────────────────────────────────────────
    navigator_node = Node(
        package='igvc_lane_detection',
        executable='navigation_node',
        name='igvc_navigator',
        output='screen',
        remappings=[('/odom', odom_topic)],
        parameters=[
            shared_cfg,
            PathJoinSubstitution([bringup, 'config', 'navigator_config.yaml']),
            {
                'use_sim_time': use_sim_time,
                'gps_enabled': False,
                'force_identity_map_to_odom': True,
                'follow_path_enabled': True,
                'nav_strategy': 'local_lane',
                # navigator_config.yaml is shared with non-sim flows and has
                # strict 0.1 s freshness gates. In this Isaac nav-test stack,
                # /lane_costmap is published at 5 Hz (0.2 s), so relax gates
                # here in the final override layer.
                'max_costmap_age_sec': 5.0,
                'max_odom_age_sec': 0.5,
                'max_odom_costmap_skew_sec': 5.0,
            },
        ],
    )

    # ── Sim startup helper (waits for robot spawn, then starts sim) ────────
    sim_startup_helper_node = Node(
        package='igvc_test_bringup',
        executable='sim_startup_helper',
        name='sim_startup_helper',
        output='screen',
        parameters=[{
            'use_sim_time': use_sim_time,
            'robot_entity_name': robot_entity_name,
            'max_wait_seconds': 30.0,
        }],
    )

    # ── Launch description ────────────────────────────────────────────────
    return LaunchDescription([
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='true',
            description='Use simulation clock.',
        ),
        DeclareLaunchArgument(
            'track_file',
            default_value=default_track_file,
            description='Path to track_points.json from IGVC_track_generator.',
        ),
        DeclareLaunchArgument(
            'track_image_file',
            default_value=default_track_image,
            description='Path to track.png. Used for variable-width GT grid boundaries.',
        ),
        DeclareLaunchArgument(
            'track_image_pixels_per_meter',
            default_value='52.5',
        ),
        DeclareLaunchArgument(
            'debug_png_path',
            default_value='/tmp/igvc_nav_ground_truth.png',
            description='Optional PNG export path for the ground-truth occupancy grid.',
        ),
        DeclareLaunchArgument(
            'field_usd_path',
            default_value='',
            description='Optional field USD path. Empty uses simulation_interface defaults.',
        ),
        DeclareLaunchArgument(
            'robot_usd_path',
            default_value='',
            description='Optional robot USD path. Empty uses simulation_interface defaults.',
        ),
        DeclareLaunchArgument(
            'robot_entity_name',
            default_value='igvc_robot',
        ),
        DeclareLaunchArgument(
            'robot_spawn_x',
            default_value=auto_value,
            description='Robot spawn X. Default: read from track_points.json.',
        ),
        DeclareLaunchArgument(
            'robot_spawn_y',
            default_value=auto_value,
        ),
        DeclareLaunchArgument(
            'robot_spawn_z',
            default_value=auto_value,
        ),
        DeclareLaunchArgument(
            'robot_spawn_yaw_rad',
            default_value=auto_value,
        ),
        DeclareLaunchArgument(
            'odom_topic',
            default_value='/front_zed_camera_x/zed_node/odom',
            description='Odometry topic published by Isaac Sim for nav-test consumers.',
        ),
        OpaqueFunction(function=apply_robot_pose_from_json),
        sim_interface,
        sim_startup_helper_node,
        # motor_controllers,
        # twist_stamper_node,
        # odom_tf_bridge_node,
        # gt_node,
        # gt_bridge_node,
        # localization_node,
        # nav2,
        # navigator_node,
    ])
