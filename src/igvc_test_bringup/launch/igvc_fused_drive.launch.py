from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition, UnlessCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import EnvironmentVariable, LaunchConfiguration, PathJoinSubstitution, PythonExpression
from launch_ros.substitutions import FindPackageShare
from launch_ros.actions import Node



def generate_launch_description() -> LaunchDescription:
    bringup = FindPackageShare('igvc_test_bringup')
    yolo_ros_pkg = FindPackageShare('yolo_bringup')
    ublox_gps_pkg = FindPackageShare('ublox_gps')
    lidar_pkg = FindPackageShare('sllidar_ros2')

    use_sim_time = LaunchConfiguration('use_sim_time')
    gps_enabled = LaunchConfiguration('gps_enabled')
    hardware_interface = LaunchConfiguration('hardware_interface')
    sim_camera_ports = LaunchConfiguration('sim_camera_ports')
    sim_camera_address = LaunchConfiguration('sim_camera_address')
    model_weights = LaunchConfiguration('model_weights')
    is_isaac_drive = PythonExpression([
        "'", hardware_interface, "' == 'IsaacDriveHardware'"
    ])
    

    zed_multi_fused_odom = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([bringup, 'launch', 'zed_multi_fused_odom.launch.py'])
        ),
        condition=UnlessCondition(is_isaac_drive),
        launch_arguments={
            'use_sim_time': use_sim_time,
        }.items(),
    )

    zed_multi_sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([bringup, 'launch', 'zed_multi_sim.launch.py'])
        ),
        condition=IfCondition(is_isaac_drive),
        launch_arguments={
            'sim_ports': sim_camera_ports,
            'sim_address': sim_camera_address,
        }.items(),
    )

    motor_controllers = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([bringup, 'launch', 'motor_controllers.launch.py'])
        ),
        launch_arguments={
            'hardware_interface': hardware_interface,
            'use_sim_time': use_sim_time,
        }.items(),
    )
    teleop = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([bringup, 'launch', 'teleop.launch.py'])
        ),
        launch_arguments={
            'hardware_interface': hardware_interface,
            'use_sim_time': use_sim_time,
        }.items(),
    )
    lane_segmentation = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([bringup, 'launch', 'lane_segmentation.launch.py'])
        ),
        launch_arguments={
            'gps_enabled': gps_enabled,
            'force_identity_map_to_odom': 'true',
            'use_sim_time': use_sim_time,
            'model_weights': model_weights,
        }.items(),
    )

    lane_follower = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([bringup, 'launch', 'lane_follower.launch.py'])
        ),
        launch_arguments={
            'gps_enabled': gps_enabled,
            'use_sim_time': use_sim_time,
        }.items(),
    )

    # zed_f9p_launch = IncludeLaunchDescription(
    #     PythonLaunchDescriptionSource(
    #         PathJoinSubstitution(
    #             [ublox_gps_pkg, 'launch', 'ublox_gps_node_zedf9p-launch.py']
    #         )
    #     ),
    #     condition=IfCondition(gps_enabled),
    # )

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

     # Bridge Nav2 Twist output to the stamped cmd_vel expected by ros2_control.
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
    yolo_ros = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([yolo_ros_pkg, 'launch', 'yolo.launch.py'])
        ),
        launch_arguments={
            'use_sim_time': use_sim_time,
            'namespace': 'yolo_ros',
            'use_3d': 'True',
            'model': '/home/nitin-5090/Documents/DevEnv/jazzy_ws/IGVC_robot_2026/models/yolov26n.pt',
            'target_frame': 'map',
            'input_image_topic': '/front_zed_camera_x/zed_node/rgb/color/rect/image',
            'input_depth_topic': '/front_zed_camera_x/zed_node/depth/depth_registered',
            'input_depth_info_topic': '/front_zed_camera_x/zed_node/depth/camera_info',
            'depth_image_units_divisor': '1',
        }.items(),
    )
    # YOLO bounding-box obstacle costmap disabled — large merged boxes were
    # covering the entire track.  Replaced by lidar + depth-based costmaps.
    # object_detection_to_costmap_node = Node(
    #     package='igvc_lane_detection',
    #     executable='obstacle_costmap_node',
    #     name='obstacle_costmap_node',
    #     output='screen',
    #     parameters=[
    #         {'use_sim_time': use_sim_time},
    #         {'frame_id': 'odom'},
    #         {'width_m': 100.0},
    #         {'height_m': 100.0},
    #         {'resolution': 0.2},
    #         {'obstacle_lifetime_sec': 300.0},
    #         {'detections_topic': '/yolo_ros/detections_3d'},
    #     ],
    # )

    depth_obstacle_costmap_node = Node(
        package='igvc_lane_detection',
        executable='depth_obstacle_costmap_node',
        name='depth_obstacle_costmap_node',
        output='screen',
        parameters=[
            {'use_sim_time': use_sim_time},
            {'frame_id': 'odom'},
            {'depth_topic': '/front_zed_camera_x/zed_node/depth/depth_registered'},
            {'info_topic': '/front_zed_camera_x/zed_node/depth/camera_info'},
            {'output_topic': '/depth_obstacle_map'},
            {'use_odom_pose': True},
            {'odom_topic': '/front_zed_camera_x/zed_node/odom'},
            {'camera_x_m': 0.44},
            {'camera_y_m': 0.06},
            {'camera_z_m': 0.21},
            {'camera_roll_rad': 0.0},
            {'camera_pitch_rad': 0.0},
            {'camera_yaw_rad': 0.0},
            {'width_m': 100.0},
            {'height_m': 100.0},
            {'origin_x': -50.0},
            {'origin_y': -50.0},
            {'resolution': 0.10},
            {'min_depth_m': 0.40},
            {'max_depth_m': 5.5},
            {'min_height_m': 0.25},
            {'max_height_m': 2.20},
            {'stride': 6},
            {'hit_weight': 1.0},
            {'decay': 0.97},
            {'decay_every_n_frames': 1},
            {'hit_threshold': 14.0},
            {'min_points_per_cell': 2},
            {'min_component_cells': 8},
            {'inflate_radius_m': 0.10},
            {'publish_hz': 5.0},
            {'use_latest_tf': True},
            {'max_frame_age_sec': 2.0},
            {'depth_frame_convention': 'auto'},
        ],
    )
    lidar_obstacle_costmap_node = Node(
        package='igvc_lane_detection',
        executable='lidar_obstacle_costmap_node',
        name='lidar_obstacle_costmap_node',
        output='screen',
        parameters=[
            {'use_sim_time': use_sim_time},
            {'frame_id': 'odom'},
            {'scan_topic': '/scan'},
            {'output_topic': '/lidar_obstacle_map'},
            {'width_m': 100.0},
            {'height_m': 100.0},
            {'origin_x': -50.0},
            {'origin_y': -50.0},
            {'resolution': 0.10},
            {'min_range_m': 0.15},
            {'max_range_m': 10.0},
            {'hit_weight': 5.0},
            {'free_weight': 0.5},
            {'hit_threshold': 4.0},
            {'free_threshold': 3.0},
            {'inflate_radius_m': 0.20},
            {'publish_hz': 5.0},
        ],
    )


    return LaunchDescription([
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='false',
            description='Use simulation clock if true.',
        ),
        DeclareLaunchArgument(
            'gps_enabled',
            default_value='false',
            choices=['true', 'false'],
            description='Enable GPS-driven localization logic if true.',
        ),
        DeclareLaunchArgument(
            'hardware_interface',
            default_value='CanInterface',
            description='Hardware interface used by motor controllers.',
        ),
        DeclareLaunchArgument(
            'model_weights',
            default_value=EnvironmentVariable('LANE_MODEL_WEIGHTS', default_value=''),
            description=(
                'Absolute path to the selected lane model checkpoint. If empty, '
                'lane_segmentation_node chooses $UFLDV2_WEIGHTS or $YOLOPV2_WEIGHTS '
                'from detection_mode.')
        ),
        # zed_multi_fused_odom,
        # teleop,
        motor_controllers,
        # lane_follower,
        lane_segmentation,
        # gps_node,
        odom_tf_bridge_node,
        # zed_f9p_launch,
        twist_stamper_node,
        # object_detection_to_costmap_node,  # disabled: YOLO bbox obstacles
        # yolo_ros,  # disabled: only fed obstacle_costmap_node
        # lidar_obstacle_costmap_node,  # disabled: obstacles shorter than lidar plane
        depth_obstacle_costmap_node,
    ])