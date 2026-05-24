from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare
from launch_ros.actions import Node



def generate_launch_description() -> LaunchDescription:
    bringup = FindPackageShare('igvc_test_bringup')
    yolo_ros_pkg = FindPackageShare('yolo_bringup')
    ublox_gps_pkg = FindPackageShare('ublox_gps')

    use_sim_time = LaunchConfiguration('use_sim_time')
    gps_enabled = LaunchConfiguration('gps_enabled')
    hardware_interface = LaunchConfiguration('hardware_interface')
    gps_config = PathJoinSubstitution([bringup, 'config', 'zed_f9p.yaml'])
    

    zed_multi_fused_odom = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([bringup, 'launch', 'zed_multi_fused_odom.launch.py'])
        ),
        launch_arguments={
            'use_sim_time': use_sim_time,
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
    gps_node = Node(
        package='ublox_gps',
        executable='ublox_gps_node',
        name='ublox_gps_node',
        output='screen',
        parameters=[
            {'use_sim_time': use_sim_time},
            {'publish_odom': True},
            # {'publish_tf': True},
            gps_config,
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
            'use_3d': 'True',
            'model': '/home/nitin/Documents/DevEnv/jazzy_ws/IGVC_robot_2026/weights/yolov26n.pt',
            'target_link': 'map',
            'imgsz_height': '640',
            'imgsz_width': '640',
            'input_image_topic': '/front_zed_camera_x/zed_node/rgb/image_rect_color',
            'camera_info_topic': '/front_zed_camera_x/zed_node/rgb/camera_info',
            'input_depth_topic': '/front_zed_camera_x/zed_node/depth/depth_registered',
        }.items(),
    )
    object_detection_to_costmap_node = Node(
        package='igvc_lane_detection',
        executable='obstacle_costmap_node',
        name='obstacle_costmap_node',
        output='screen',
        parameters=[
            {'use_sim_time': use_sim_time},
            {'costmap_topic': '/obstacle_map'},
            {'costmap_frame_id': 'map'},
            {'costmap_size_x': 10.0},
            {'costmap_size_y': 10.0},
            {'costmap_resolution': 0.1},
            {'detection_timeout_sec': 1.0},
        ],
        remappings=[
            ('yolo_detections', '/yolo_ros/detections'),
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
        # zed_multi_fused_odom,
        # teleop,
        motor_controllers,
        # lane_follower,
        lane_segmentation,
        # gps_node,
        odom_tf_bridge_node,
        # zed_f9p_launch,
        twist_stamper_node,
        object_detection_to_costmap_node,
        yolo_ros
    ])