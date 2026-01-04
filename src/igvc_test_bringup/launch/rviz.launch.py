import os

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.substitutions import LaunchConfiguration, Command
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch_ros.actions import Node

import xacro

def generate_launch_description():

    # Check if we're told to use sim time
    hardware_interface = LaunchConfiguration('hardware_interface')
    use_sim_time = LaunchConfiguration('use_sim_time')

    # Process the URDF file
    pkg_path = os.path.join(get_package_share_directory('igvc_test_description'))
    xacro_file = os.path.join(pkg_path,'urdf','robots','test_robot.urdf.xacro')
    # robot_description_config = xacro.process_file(xacro_file).toxml()
    robot_description_config = Command(['xacro ', xacro_file, ' hardware_interface:=', hardware_interface])
    
    # Create a robot_state_publisher node
    params = {'robot_description': robot_description_config, 'use_sim_time': use_sim_time}
    node_robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='screen',
        parameters=[params]
    )

    joint_state_publisher_gui = Node(
        package='joint_state_publisher_gui',
        executable='joint_state_publisher_gui',
        output='screen'
    )

    # Start Rviz2 with basic view
    rviz2_config_path = os.path.join(get_package_share_directory('igvc_test_bringup'), 'config/config.rviz')
    run_rviz2 = ExecuteProcess(
        cmd=['rviz2', '-d', rviz2_config_path],
        output='screen'
    )
    # Launch!
    return LaunchDescription([
        DeclareLaunchArgument(
            'hardware_interface',
            default_value='IsaacDriveHardware',
            description='Define what hardware interface to use'),
        
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='true',
            description='Use sim time if true'),

        node_robot_state_publisher,
        joint_state_publisher_gui,
        run_rviz2
    ])
