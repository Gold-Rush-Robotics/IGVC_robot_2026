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
    disable_tf = LaunchConfiguration('disable_tf')

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
            'disable_tf': disable_tf,
        }.items(),
    )

    ekf_node = Node(
        package='robot_localization',
        executable='ekf_node',
        name='zed_multi_ekf',
        output='screen',
        parameters=[
            PathJoinSubstitution([bringup, 'config', 'zed_multi_ekf.yaml']),
            {
                'use_sim_time': use_sim_time,
                'odom0': LaunchConfiguration('odom0_topic'),
                'odom1': LaunchConfiguration('odom1_topic'),
                'odom2': LaunchConfiguration('odom2_topic'),
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
            default_value='[43593214,40636496,46941578]',
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
            'disable_tf',
            default_value='true',
            description='Disable ZED TF publishing and let EKF own odom->base_link TF.',
        ),
        DeclareLaunchArgument(
            'odom0_topic',
            default_value='/left_zed_camera_x/zed_node/odom',
            description='Left camera odom topic.',
        ),
        DeclareLaunchArgument(
            'odom1_topic',
            default_value='/front_zed_camera_x/zed_node/odom',
            description='Front camera odom topic.',
        ),
        DeclareLaunchArgument(
            'odom2_topic',
            default_value='/right_zed_camera_x/zed_node/odom',
            description='Right camera odom topic.',
        ),
        zed_multi_launch,
        ekf_node,
    ])