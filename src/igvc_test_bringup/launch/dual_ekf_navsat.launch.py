"""
dual_ekf_navsat.launch.py

robot_localization GPS-localization stack for Nav2 GPS waypoint following.
Mirrors nav2_gps_waypoint_follower_demo/launch/dual_ekf_navsat.launch.py,
adapted to the IGVC robot's sensor topics.

Starts:
  1. ekf_filter_node_odom  — local EKF, publishes  odom -> base_link TF
  2. ekf_filter_node_map   — global EKF, publishes map  -> odom TF (fuses GPS)
  3. navsat_transform      — /fix (+ heading) -> /odometry/gps (map frame)

Params: igvc_test_bringup/config/dual_ekf_navsat_params.yaml

Arguments
---------
    use_sim_time    true | false                              (default: false)
    params_file     robot_localization YAML params            (default: dual_ekf_navsat_params.yaml)
    gps_topic       NavSatFix input topic                     (default: /fix)
    imu_topic       absolute-heading Imu input topic          (default: /front_zed_camera_2i/imu/heading)
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description() -> LaunchDescription:
    bringup = FindPackageShare('igvc_test_bringup')

    use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time', default_value='false',
        description='Use /clock from a simulator.')
    params_file_arg = DeclareLaunchArgument(
        'params_file',
        default_value=PathJoinSubstitution(
            [bringup, 'config', 'dual_ekf_navsat_params.yaml']),
        description='robot_localization params YAML.')
    gps_topic_arg = DeclareLaunchArgument(
        'gps_topic', default_value='/fix',
        description='NavSatFix topic fed to navsat_transform.')
    imu_topic_arg = DeclareLaunchArgument(
        'imu_topic', default_value='/front_zed_camera_2i/imu/heading',
        description='Absolute-heading Imu topic fed to navsat_transform.')

    use_sim_time = LaunchConfiguration('use_sim_time')
    params_file = LaunchConfiguration('params_file')
    gps_topic = LaunchConfiguration('gps_topic')
    imu_topic = LaunchConfiguration('imu_topic')

    ekf_odom = Node(
        package='robot_localization',
        executable='ekf_node',
        name='ekf_filter_node_odom',
        output='screen',
        parameters=[params_file, {'use_sim_time': use_sim_time}],
        remappings=[('odometry/filtered', 'odometry/local')],
    )

    ekf_map = Node(
        package='robot_localization',
        executable='ekf_node',
        name='ekf_filter_node_map',
        output='screen',
        parameters=[params_file, {'use_sim_time': use_sim_time}],
        remappings=[('odometry/filtered', 'odometry/global')],
    )

    navsat_transform = Node(
        package='robot_localization',
        executable='navsat_transform_node',
        name='navsat_transform',
        output='screen',
        parameters=[params_file, {'use_sim_time': use_sim_time}],
        remappings=[
            ('imu/data', imu_topic),
            ('gps/fix', gps_topic),
            ('gps/filtered', 'gps/filtered'),
            ('odometry/gps', 'odometry/gps'),
            ('odometry/filtered', 'odometry/global'),
        ],
    )

    return LaunchDescription([
        use_sim_time_arg,
        params_file_arg,
        gps_topic_arg,
        imu_topic_arg,
        ekf_odom,
        ekf_map,
        navsat_transform,
    ])
