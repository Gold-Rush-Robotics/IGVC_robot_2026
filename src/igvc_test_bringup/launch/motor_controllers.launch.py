import os
from ament_index_python.packages import get_package_prefix, get_package_share_directory
from launch import LaunchDescription
from launch.actions import ExecuteProcess, IncludeLaunchDescription, RegisterEventHandler
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
import xacro

def generate_launch_description():

    # Get Local Files
    description_pkg_path = os.path.join(get_package_share_directory('igvc_test_description'))
    config_pkg_path = os.path.join(get_package_share_directory('igvc_test_bringup'))
    joystick_file = os.path.join(config_pkg_path, 'config', 'xbox-holonomic.config.yaml')
    xacro_file = os.path.join(description_pkg_path, 'urdf', 'robots','test_robot.urdf.xacro')
    controllers_file = os.path.join(config_pkg_path, 'config', 'controllers.yaml')
    rviz_file = os.path.join(config_pkg_path, 'config', 'config.rviz')
    robot_description_config = xacro.process_file(xacro_file)
    robot_description_xml = robot_description_config.toxml()


    description_params = {'robot_description': robot_description_xml, 'use_sim_time': False }
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
        parameters=[{'use_sim_time': False }, controllers_file],
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


    # Start Rviz2 with basic view
    run_rviz2_node = Node(
        package='rviz2',
        executable='rviz2',
        parameters=[{ 'use_sim_time': False }],
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


    # Start Joystick Node
    joy = Node(
            package='joy', 
            executable='joy_node', 
            name='joy_node',
            parameters=[{
                'dev': '/dev/input/js0',
                'deadzone': 0.3,
                'autorepeat_rate': 20.0,
            }])


    # Start Teleop Node to translate joystick commands to robot commands
    joy_teleop = Node(
        package='teleop_twist_joy', 
        executable='teleop_node',
        name='teleop_twist_joy_node', 
        parameters=[joystick_file],
        remappings={('/cmd_vel', '/diff_drive_controller/cmd_vel')}
        )


    # Launch!
    return LaunchDescription([
        control_node,
        node_robot_state_publisher,
        joint_state_broadcaster_spawner,
        diff_drive_spawner,
        # rviz2_delay,
        joy,
        joy_teleop
    ])