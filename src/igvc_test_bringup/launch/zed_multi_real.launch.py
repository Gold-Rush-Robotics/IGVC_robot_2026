from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def generate_launch_description() -> LaunchDescription:
    bringup = FindPackageShare('igvc_test_bringup')

    fused_odom = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([bringup, 'launch', 'zed_multi_fused_odom.launch.py'])
        ),
        launch_arguments={
            'cam_names': LaunchConfiguration('cam_names'),
            'cam_models': LaunchConfiguration('cam_models'),
            'cam_serials': LaunchConfiguration('cam_serials'),
            'cam_ids': '[]',
            'sim_ports': '[]',
            'namespace': LaunchConfiguration('namespace'),
            'use_sim_time': 'false',
            'sim_mode': 'false',
            'sim_address': '',
            'disable_tf': LaunchConfiguration('disable_tf'),
            'ros_params_override_path': PathJoinSubstitution(
                [bringup, 'config', 'common_stereo_real.yaml']
            ),
        }.items(),
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
            description='Physical serial numbers array: left, front, right.',
        ),
        DeclareLaunchArgument(
            'namespace',
            default_value='',
            description='Optional top-level namespace.',
        ),
        DeclareLaunchArgument(
            'disable_tf',
            default_value='true',
            description='Let robot_localization own odom->base_link TF by default.',
        ),
        fused_odom,
    ])
