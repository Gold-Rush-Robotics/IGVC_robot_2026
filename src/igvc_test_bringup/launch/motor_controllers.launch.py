import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, IncludeLaunchDescription, RegisterEventHandler
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command, LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():

    # Declare launch arguments
    hardware_interface_arg = DeclareLaunchArgument(
        'hardware_interface',
        default_value='CanInterface',
        description='Hardware interface to use (IsaacDriveHardware or CanInterface)'
    )
    use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time',
        default_value='false',
        description='Use simulation clock if true'
    )
    hardware_interface = LaunchConfiguration('hardware_interface')
    use_sim_time = LaunchConfiguration('use_sim_time')

    # Get Local Files
    description_pkg_path = os.path.join(get_package_share_directory('igvc_test_description'))
    config_pkg_path = os.path.join(get_package_share_directory('igvc_test_bringup'))
    xacro_file = os.path.join(description_pkg_path, 'urdf', 'robots','test_robot.urdf.xacro')
    controllers_file = os.path.join(config_pkg_path, 'config', 'controllers.yaml')
    rviz_file = os.path.join(config_pkg_path, 'config', 'config.rviz')

    robot_description = Command([
        'xacro ', xacro_file,
        ' hardware_interface:=', hardware_interface
    ])


    description_params = {'robot_description': robot_description, 'use_sim_time': use_sim_time }
    node_robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='screen',
        parameters=[description_params]
    )


    # Starts ROS2 Control
    control_node = Node(
        package="controller_manager",
        executable="ros2_control_node",
        parameters=[description_params, controllers_file],
        remappings=[('~/robot_description', '/robot_description')],
        output="screen",
    )


    # Starts ROS2 Control Joint State Broadcaster
    joint_state_broadcaster_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["joint_state_broadcaster", "--controller-manager", "/controller_manager"],
    )
    


    diff_drive_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["diff_drive_controller", "--controller-manager", "/controller_manager"],
    )

    # ── GPIO / operator-interface nodes ───────────────────────────────────
    # Blinks pin 13 at 2 Hz in autonomous mode; solid ON otherwise.
    autonomous_indicator_node = Node(
        package='igvc_lane_detection',
        executable='autonomous_indicator_node',
        name='autonomous_indicator_node',
        output='screen',
    )

    # Applies brakes (pins 18+22 HIGH) on B button, releases on A button.
    brake_control_node = Node(
        package='igvc_lane_detection',
        executable='brake_control_node',
        name='brake_control_node',
        output='screen',
    )


    # Start Rviz2 with basic view
    run_rviz2_node = Node(
        package='rviz2',
        executable='rviz2',
        parameters=[{ 'use_sim_time': use_sim_time }],
        name='isaac_rviz2',
        output='screen',
        arguments=[["-d"], [rviz_file]],
    )

    rviz2_delay = RegisterEventHandler(
        event_handler=OnProcessExit(
            target_action=joint_state_broadcaster_spawner,
            on_exit=[run_rviz2_node],
        )
    )


    # Launch!
    return LaunchDescription([
        hardware_interface_arg,
        use_sim_time_arg,
        control_node,
        node_robot_state_publisher,
        joint_state_broadcaster_spawner,
        diff_drive_spawner,
        autonomous_indicator_node,
        brake_control_node,
        # rviz2_delay,
    ])