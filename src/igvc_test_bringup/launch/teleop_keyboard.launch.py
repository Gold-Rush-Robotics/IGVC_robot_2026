import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    hardware_interface = LaunchConfiguration('hardware_interface')
    use_sim_time = LaunchConfiguration('use_sim_time')

    config_pkg_path = get_package_share_directory('igvc_test_bringup')
    mux_config = os.path.join(config_pkg_path, 'config', 'twist_mux.yaml')

    node_mux = Node(
        package = 'twist_mux',
        executable = 'twist_mux',
        parameters = [mux_config, {'use_sime_time': use_sim_time}],
        remappings = [("cmd_vel_out", '/diff_drive_controller/cmd_vel')],
        output = 'screen'
    )

    node_keyboard = Node(
        package = 'teleop_twist_keyboard',
        executable = 'teleop_twist_keyboard',
        parameters = [{'stamped': True, 'use_sim_time': use_sim_time}],
        remappings = [('/cmd_vel', '/cmd_vel_keyboard')],
        prefix = 'xterm -e',
        output = 'screen'
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            'hardware_interface',
            default_value='GazeboSimSystem',
            description='Hardware interface to use (GazeboSimSystem or CanInterface)'
        ),
        
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='true',
            description='Use simulation clock if true'
        ),

        node_mux,
        node_keyboard
    ])
