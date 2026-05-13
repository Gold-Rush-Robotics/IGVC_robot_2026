from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def generate_launch_description() -> LaunchDescription:
    bringup = FindPackageShare('igvc_test_bringup')

    zed_multi = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([bringup, 'launch', 'zed_multi.launch.py'])
        ),
        launch_arguments={
            'cam_names': LaunchConfiguration('cam_names'),
            'cam_models': LaunchConfiguration('cam_models'),
            'cam_serials': LaunchConfiguration('cam_serials'),
            'cam_ids': '[]',
            'sim_ports': LaunchConfiguration('sim_ports'),
            'namespace': LaunchConfiguration('namespace'),
            'use_sim_time': 'true',
            'sim_mode': 'true',
            'sim_address': LaunchConfiguration('sim_address'),
            'disable_tf': 'true',
            'ros_params_override_path': PathJoinSubstitution(
                [bringup, 'config', 'common_stereo_sim.yaml']
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
            description='Virtual/physical serial numbers array.',
        ),
        DeclareLaunchArgument(
            'sim_ports',
            default_value='[30000,30001,30002]',
            description='Isaac Sim ZED stream ports: left, front, right.',
        ),
        DeclareLaunchArgument(
            'sim_address',
            default_value='',
            description='Optional Isaac Sim host address. Leave empty to use common_stereo_sim.yaml.',
        ),
        DeclareLaunchArgument(
            'namespace',
            default_value='',
            description='Optional top-level namespace.',
        ),
        zed_multi,
    ])
