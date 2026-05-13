from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description() -> LaunchDescription:
    bringup = FindPackageShare('igvc_test_bringup')

    cam_names = LaunchConfiguration('cam_names')
    cam_models = LaunchConfiguration('cam_models')
    cam_serials = LaunchConfiguration('cam_serials')
    cam_ids = LaunchConfiguration('cam_ids')
    sim_ports = LaunchConfiguration('sim_ports')
    namespace = LaunchConfiguration('namespace')
    use_sim_time = LaunchConfiguration('use_sim_time')
    sim_mode = LaunchConfiguration('sim_mode')
    sim_address = LaunchConfiguration('sim_address')
    disable_tf = LaunchConfiguration('disable_tf')
    ros_params_override_path = LaunchConfiguration('ros_params_override_path')

    zed_multi_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([bringup, 'launch', 'zed_multi.launch.py'])
        ),
        launch_arguments={
            'cam_names': cam_names,
            'cam_models': cam_models,
            'cam_serials': cam_serials,
            'cam_ids': cam_ids,
            'sim_ports': sim_ports,
            'namespace': namespace,
            'use_sim_time': use_sim_time,
            'sim_mode': sim_mode,
            'sim_address': sim_address,
            'disable_tf': disable_tf,
            'ros_params_override_path': ros_params_override_path,
        }.items(),
    )

    # Add these remappings so all cameras are in the same odom frame
    ekf_node = Node(
        package='robot_localization',
        executable='ekf_node',
        name='zed_multi_ekf',
        output='screen',
        remappings=[
            ('/odometry/filtered', '/odom'),  # standardize output topic
        ],
        parameters=[
            PathJoinSubstitution([bringup, 'config', 'zed_multi_ekf.yaml']),
            {
                'use_sim_time': use_sim_time,
            },
        ],
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            'cam_names',
            default_value='[left_zed_camera_x,front_zed_camera_x,right_zed_camera_x]',
            description='Camera names array.',
        ),
        DeclareLaunchArgument(
            'cam_models',
            default_value='[zedx,zedx,zedx]',
            description='Camera models array.',
        ),
        DeclareLaunchArgument(
            'cam_serials',
            default_value='[46941578,40636496,43593214]',
            description='Optional camera serials array (or empty).',
        ),
        DeclareLaunchArgument(
            'cam_ids',
            default_value='[]',
            description='Optional camera IDs array (or empty).',
        ),
        DeclareLaunchArgument(
            'sim_ports',
            default_value='[]',
            description='Optional simulation ports array (or empty).',
        ),
        DeclareLaunchArgument(
            'namespace',
            default_value='',
            description='Optional top-level namespace.',
        ),
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='false',
            description='Use simulation clock if true.',
        ),
        DeclareLaunchArgument(
            'sim_mode',
            default_value='false',
            description='Enable ZED simulation mode if true.',
        ),
        DeclareLaunchArgument(
            'sim_address',
            default_value='',
            description='Optional simulation server address. Leave empty to use the YAML default.',
        ),
        DeclareLaunchArgument(
            'disable_tf',
            default_value='true',
            description='Disable ZED TF publishing and let EKF own odom->base_link TF.',
        ),
        DeclareLaunchArgument(
            'ros_params_override_path',
            default_value=PathJoinSubstitution([bringup, 'config', 'common_stereo_real.yaml']),
            description='YAML file that overrides ZED wrapper defaults for every camera. '
                        'Defaults to igvc_test_bringup/config/common_stereo_real.yaml.',
        ),
        zed_multi_launch,
        ekf_node,
    ])