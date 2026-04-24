from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare
from launch_ros.actions import Node



def generate_launch_description() -> LaunchDescription:
    bringup = FindPackageShare('igvc_test_bringup')

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
    lane_segmentation = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([bringup, 'launch', 'lane_segmentation.launch.py'])
        ),
        launch_arguments={
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

    odom_to_tf_ros2 = Node(
        package="odom_to_tf_ros2",
        executable="odom_to_tf",
        name="odom_to_tf",
        output="screen",
        parameters=[
            {'use_sim_time': use_sim_time},
            {'odom_topic': '/front_zed_camera_x/zed_node/odom'},
            {'frame_id': 'odom'},
            {'child_frame_id': 'base_link'},
            {'use_original_timestamp': True},
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
        motor_controllers,
        # lane_follower,
        lane_segmentation,
        odom_to_tf_ros2,
        # gps_node,
        twist_stamper_node,
    ])