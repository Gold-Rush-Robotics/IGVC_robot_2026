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

    lane_follower = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([bringup, 'launch', 'lane_follower.launch.py'])
        ),
        launch_arguments={
            'gps_enabled': gps_enabled,
            'use_sim_time': use_sim_time,
        }.items(),
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
        zed_multi_fused_odom,
        motor_controllers,
        lane_follower,
        # odom_tf_bridge_node,
    ])