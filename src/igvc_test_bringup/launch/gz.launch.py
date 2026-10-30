import os 
from launch_ros.actions import Node
from launch import LaunchDescription
from launch.substitutions import LaunchConfiguration, Command
from ament_index_python.packages import get_package_share_directory
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.parameter_descriptions import ParameterValue

def generate_launch_description():
    world = LaunchConfiguration('world')
    use_sim_time = LaunchConfiguration('use_sim_time')
    hardware_interface = LaunchConfiguration('hardware_interface')

    pkg_description_share = get_package_share_directory('igvc_test_description')
    pkg_bringup_share = get_package_share_directory('igvc_test_bringup')
    ros_gz_sim = get_package_share_directory('ros_gz_sim')

    xacro_file = os.path.join(pkg_description_share, 'urdf', 'robots', 'test_robot.urdf.xacro')
    robot_description_config = ParameterValue(
        Command(['xacro ', xacro_file, ' hardware_interface:=', hardware_interface]),
        value_type = str,
    )

    node_gz_sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(ros_gz_sim, 'launch', 'gz_sim.launch.py')),
        launch_arguments = {'gz_args': ['-r -v 4 ', world], 'on_exit_shutdown': 'true'}.items()
    )

    rsp_params = {'robot_description': robot_description_config, 'use_sim_time': use_sim_time}
    node_rsp = Node(
        package = 'robot_state_publisher',
        executable = 'robot_state_publisher',
        output = 'screen',
        parameters = [rsp_params]
    )

    node_rviz2 = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([os.path.join(pkg_bringup_share, 'launch', 'rviz.launch.py')])
    )

    node_spawn = Node(
        package = 'ros_gz_sim',
        executable = 'create',
        arguments = ['-topic', 'robot_description', '-name', 'igvc_test_bot', '-z', '0.25'],
        output = 'screen'
    )

    bridge_params = os.path.join(pkg_bringup_share, 'config', 'gz_bridge.yaml')
    node_gz_bridge = Node(
        package = 'ros_gz_bridge',
        executable = 'parameter_bridge',
        arguments = ['--ros-args', '-p', f'config_file:={bridge_params}'],
        output = 'screen'
    )

    node_diff_spawner = Node(
        package = 'controller_manager',
        executable = 'spawner',
        arguments = ['diff_drive_controller',
                     '--param-file', os.path.join(pkg_bringup_share, 'config', 'controllers.yaml'),
                     '--controller-manager-timeout', '90']
    )

    node_joint_broad_spawner = Node(
        package = 'controller_manager',
        executable = 'spawner',
        arguments = ['joint_state_broadcaster', '--controller-manager-timeout', '90']
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            'world',
            default_value = os.path.join(pkg_description_share, 'worlds', 'obstacles_primitives.sdf'),
            description = 'Gazebo world file'
        ),

        DeclareLaunchArgument(
            'use_sim_time',
            default_value = 'true',
            description = 'Use sim time by default'
        ),

        DeclareLaunchArgument(
            'hardware_interface',
            default_value = 'GazeboSimSystem',
            description = 'The hardware interface to use for controlling the robot'
        ),

        node_gz_sim,
        node_rsp,
        node_spawn,
        node_rviz2,
        node_gz_bridge,
        node_diff_spawner,
        node_joint_broad_spawner
    ])