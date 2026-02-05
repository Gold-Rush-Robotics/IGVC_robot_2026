"""Minimal Python launch to publish robot_description from xacro, then include YAML launch."""
import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import FrontendLaunchDescriptionSource
from launch_ros.actions import Node
import xacro


def generate_launch_description():
    # Process xacro to get robot_description
    description_pkg = get_package_share_directory('igvc_test_description')
    bringup_pkg = get_package_share_directory('igvc_test_bringup')
    
    xacro_file = os.path.join(description_pkg, 'urdf', 'robots', 'test_robot.urdf.xacro')
    robot_description_xml = xacro.process_file(xacro_file).toxml()

    # Robot State Publisher with processed xacro
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='screen',
        parameters=[{
            'robot_description': robot_description_xml,
            'use_sim_time': False
        }]
    )

    # Include the YAML launch for everything else
    yaml_launch = IncludeLaunchDescription(
        FrontendLaunchDescriptionSource(
            os.path.join(bringup_pkg, 'launch', 'motor_controllers_nodes.launch.yaml')
        )
    )

    return LaunchDescription([
        robot_state_publisher,
        yaml_launch,
    ])
