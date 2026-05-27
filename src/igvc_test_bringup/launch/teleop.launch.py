import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():

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

    config_pkg_path = os.path.join(get_package_share_directory('igvc_test_bringup'))
    joystick_file = os.path.join(config_pkg_path, 'config', 'xbox-holonomic.config.yaml')
    motor_controllers_launch = os.path.join(config_pkg_path, 'launch', 'motor_controllers.launch.py')

    motor_controllers = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(motor_controllers_launch),
        launch_arguments={
            'hardware_interface': LaunchConfiguration('hardware_interface'),
            'use_sim_time': LaunchConfiguration('use_sim_time'),
        }.items()
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
        remappings=[('/cmd_vel', '/diff_drive_controller/cmd_vel')]
    )

    return LaunchDescription([
        hardware_interface_arg,
        use_sim_time_arg,
        motor_controllers,
        joy,
        joy_teleop,
    ])
