"""Minimal Python launch to publish robot_description from xacro, then include YAML launch."""
import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import FrontendLaunchDescriptionSource
from launch.substitutions import Command, LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    # Declare launch arguments
    hardware_interface_arg = DeclareLaunchArgument(
        'hardware_interface',
        default_value='IsaacDriveHardware',
        description='Hardware interface to use (IsaacDriveHardware or CanInterface)'
    )
    use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time',
        default_value='false',
        description='Use simulation clock if true'
    )
    
    # Get launch configurations
    hardware_interface = LaunchConfiguration('hardware_interface')
    use_sim_time = LaunchConfiguration('use_sim_time')
    
    # Get package paths
    description_pkg = get_package_share_directory('igvc_test_description')
    bringup_pkg = get_package_share_directory('igvc_test_bringup')
    
    xacro_file = os.path.join(description_pkg, 'urdf', 'robots', 'test_robot.urdf.xacro')
    
    # Process xacro with hardware_interface argument
    robot_description = Command([
        'xacro ', xacro_file,
        ' hardware_interface:=', hardware_interface
    ])

    # Robot State Publisher with processed xacro
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='screen',
        parameters=[{
            'robot_description': robot_description,
            'use_sim_time': use_sim_time
        }]
    )

    # Include the YAML launch for everything else
    yaml_launch = IncludeLaunchDescription(
        FrontendLaunchDescriptionSource(
            os.path.join(bringup_pkg, 'launch', 'motor_controllers_nodes.launch.yaml')
        ),
        launch_arguments={'use_sim_time': use_sim_time}.items()
    )

    return LaunchDescription([
        hardware_interface_arg,
        use_sim_time_arg,
        robot_state_publisher,
        yaml_launch,
    ])
