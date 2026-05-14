from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare
from launch_ros.actions import Node



def generate_launch_description() -> LaunchDescription:
    bringup = FindPackageShare('igvc_test_bringup')
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
        teleop,
        # lane_follower,
        lane_segmentation,
        gps_node,
        odom_tf_bridge_node,
        # zed_f9p_launch,
        twist_stamper_node,
    ])