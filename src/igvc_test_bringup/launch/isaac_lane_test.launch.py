import json
import os
from pathlib import Path

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.conditions import IfCondition
from launch.launch_description_sources import AnyLaunchDescriptionSource, PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, PythonExpression
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def _resolve_workspace_root() -> str:
    candidates = []

    env_root = os.environ.get('IGVC_WORKSPACE_ROOT', '')
    if env_root:
        candidates.append(os.path.abspath(env_root))

    candidates.append(os.getcwd())

    here = os.path.dirname(__file__)
    candidates.append(os.path.abspath(os.path.join(here, '..', '..', '..')))
    candidates.append(os.path.abspath(os.path.join(here, '..', '..', '..', '..', '..')))

    for root in candidates:
        if os.path.isdir(os.path.join(root, 'IGVC_track_generator')):
            return root

    return os.path.abspath(os.path.join(here, '..', '..', '..'))


def generate_launch_description() -> LaunchDescription:
    bringup = FindPackageShare('igvc_test_bringup')
    workspace_root = _resolve_workspace_root()
    default_track_file = os.path.join(
        workspace_root, 'IGVC_track_generator', 'track_points.json')
    default_track_image = os.path.join(
        workspace_root, 'IGVC_track_generator', 'track.png')

    mode = LaunchConfiguration('mode')
    use_sim_time = LaunchConfiguration('use_sim_time')
    model_weights = LaunchConfiguration('model_weights')
    track_file = LaunchConfiguration('track_file')
    track_image_file = LaunchConfiguration('track_image_file')
    track_image_pixels_per_meter = LaunchConfiguration('track_image_pixels_per_meter')
    debug_png_path = LaunchConfiguration('debug_png_path')
    ui_enabled = LaunchConfiguration('ui_enabled')
    eval_csv_path = LaunchConfiguration('eval_csv_path')
    field_usd_path = LaunchConfiguration('field_usd_path')
    robot_usd_path = LaunchConfiguration('robot_usd_path')
    robot_entity_name = LaunchConfiguration('robot_entity_name')
    robot_spawn_x = LaunchConfiguration('robot_spawn_x')
    robot_spawn_y = LaunchConfiguration('robot_spawn_y')
    robot_spawn_z = LaunchConfiguration('robot_spawn_z')
    robot_spawn_yaw_rad = LaunchConfiguration('robot_spawn_yaw_rad')

    auto_value = '__auto__'
    fallback_x = 8.128949212924178
    fallback_y = 16.63174225312982
    fallback_z = 0.0820257550378943
    fallback_yaw = -3.141592653589793

    is_closed_loop = PythonExpression(["'", mode, "' == 'closed_loop'"])

    def apply_robot_pose_from_json(context, *args, **kwargs):
        """Extract robot_start_pose from track JSON if available and not explicitly overridden."""
        track_file_str = context.launch_configurations.get('track_file', '')
        use_auto_x = context.launch_configurations.get('robot_spawn_x', auto_value) == auto_value
        use_auto_y = context.launch_configurations.get('robot_spawn_y', auto_value) == auto_value
        use_auto_z = context.launch_configurations.get('robot_spawn_z', auto_value) == auto_value
        use_auto_yaw = context.launch_configurations.get('robot_spawn_yaw_rad', auto_value) == auto_value
        needs_auto = use_auto_x or use_auto_y or use_auto_z or use_auto_yaw

        if not needs_auto:
            return []

        def _set_auto_defaults() -> None:
            if use_auto_x:
                context.launch_configurations['robot_spawn_x'] = str(fallback_x)
            if use_auto_y:
                context.launch_configurations['robot_spawn_y'] = str(fallback_y)
            if use_auto_z:
                context.launch_configurations['robot_spawn_z'] = str(fallback_z)
            if use_auto_yaw:
                context.launch_configurations['robot_spawn_yaw_rad'] = str(fallback_yaw)

        if not track_file_str:
            _set_auto_defaults()
            return []
        
        try:
            track_path = Path(track_file_str).expanduser()
            if not track_path.exists() or track_path.suffix.lower() != '.json':
                _set_auto_defaults()
                return []
            
            payload = json.loads(track_path.read_text(encoding='utf-8'))
            if 'robot_start_pose' not in payload:
                _set_auto_defaults()
                return []
            
            pose = payload['robot_start_pose']
            if 'position_m' in pose and 'yaw_rad' in pose:
                # Spawn at world origin; grid is offset to match
                yaw = float(pose.get('yaw_rad', -3.141592653589793))
                if use_auto_x:
                    context.launch_configurations['robot_spawn_x'] = '0.0'
                if use_auto_y:
                    context.launch_configurations['robot_spawn_y'] = '0.0'
                if use_auto_z:
                    z = float(pose['position_m'].get('z', 0.0820257550378943))
                    context.launch_configurations['robot_spawn_z'] = str(z)
                if use_auto_yaw:
                    context.launch_configurations['robot_spawn_yaw_rad'] = str(yaw)
            else:
                _set_auto_defaults()
        except Exception:
            _set_auto_defaults()
        return []

    sim_cams = IncludeLaunchDescription(
        AnyLaunchDescriptionSource(
            PathJoinSubstitution([bringup, 'launch', 'simulation_launch.launch.yaml'])
        ),
    )

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
        }.items(),
    )

    lane_stack = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([bringup, 'launch', 'lane_segmentation.launch.py'])
        ),
        launch_arguments={
            'gps_enabled': 'false',
            'force_identity_map_to_odom': 'true',
            'use_sim_time': use_sim_time,
            'model_weights': model_weights,
            'publish_overlay': 'true',
        }.items(),
    )

    shared_cfg = PathJoinSubstitution([bringup, 'config', 'isaac_lane_test.yaml'])

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

    eval_node = Node(
        package='igvc_lane_detection',
        executable='lane_eval_node',
        name='lane_eval_node',
        output='screen',
        parameters=[
            shared_cfg,
            {
                'use_sim_time': use_sim_time,
                'report_csv_path': eval_csv_path,
            },
        ],
    )

    twist_node = Node(
        package='igvc_lane_detection',
        executable='midpoint_twist_test_node',
        name='midpoint_twist_test_node',
        output='screen',
        condition=IfCondition(is_closed_loop),
        parameters=[
            shared_cfg,
            {'use_sim_time': use_sim_time},
        ],
    )

    ui_node = Node(
        package='debug_gui',
        executable='lane_compare_ui',
        name='lane_compare_ui',
        output='screen',
        condition=IfCondition(ui_enabled),
        parameters=[
            shared_cfg,
            {'use_sim_time': use_sim_time},
        ],
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            'mode',
            default_value='perception_only',
            description='Test mode: perception_only | closed_loop',
        ),
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='true',
            description='Use simulation clock.',
        ),
        DeclareLaunchArgument(
            'model_weights',
            default_value='',
            description='YOLOPv2 model path for lane segmentation launch.',
        ),
        DeclareLaunchArgument(
            'track_file',
            default_value=default_track_file,
            description='Track file (.json/.yaml/.csv) with midpoint points exported by IGVC_track_generator.',
        ),
        DeclareLaunchArgument(
            'track_image_file',
            default_value=default_track_image,
            description='Track image (e.g., IGVC_track_generator/track.png). Used if track_file is missing.',
        ),
        DeclareLaunchArgument(
            'track_image_pixels_per_meter',
            default_value='52.5',
            description='Pixel-to-meter scale for track_image_file conversion.',
        ),
        DeclareLaunchArgument(
            'debug_png_path',
            default_value='/tmp/igvc_lane_ground_truth.png',
            description='Optional PNG export path for the generated ground-truth occupancy grid.',
        ),
        DeclareLaunchArgument(
            'ui_enabled',
            default_value='true',
            description='Launch custom PyQt lane comparison UI.',
        ),
        DeclareLaunchArgument(
            'eval_csv_path',
            default_value='',
            description='Optional CSV output path for frame metrics.',
        ),
        DeclareLaunchArgument(
            'field_usd_path',
            default_value='',
            description='Optional field/track USD path. Empty uses simulation_interface defaults.',
        ),
        DeclareLaunchArgument(
            'robot_usd_path',
            default_value='',
            description='Optional robot USD path. Empty uses simulation_interface defaults.',
        ),
        DeclareLaunchArgument(
            'robot_entity_name',
            default_value='igvc_robot',
            description='Entity name for spawned robot in Isaac Sim.',
        ),
        DeclareLaunchArgument(
            'robot_spawn_x',
            default_value=auto_value,
            description='Robot spawn X in world frame. Use default auto mode for value from track JSON.',
        ),
        DeclareLaunchArgument(
            'robot_spawn_y',
            default_value=auto_value,
            description='Robot spawn Y in world frame. Use default auto mode for value from track JSON.',
        ),
        DeclareLaunchArgument(
            'robot_spawn_z',
            default_value=auto_value,
            description='Robot spawn Z in world frame. Use default auto mode for value from track JSON.',
        ),
        DeclareLaunchArgument(
            'robot_spawn_yaw_rad',
            default_value=auto_value,
            description='Robot spawn yaw (radians). Use default auto mode for value from track JSON.',
        ),
        OpaqueFunction(function=apply_robot_pose_from_json),
        # sim_cams,
        sim_interface,
        lane_stack,
        gt_node,
        eval_node,
        twist_node,
        ui_node,
    ])
