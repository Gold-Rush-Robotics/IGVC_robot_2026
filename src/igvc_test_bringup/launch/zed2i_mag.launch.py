"""
zed2i_mag.launch.py

Brings up a single ZED 2i in MAGNETOMETER-ONLY mode.

Depth, point cloud, video, positional tracking, GNSS fusion, mapping,
object detection, body tracking and the stream server are all disabled via
config/zed2i_mag_only.yaml. The only sensor data published is the
magnetometer on:

    /<camera_name>/zed_node/imu/mag

Typical usage:
    ros2 launch igvc_test_bringup zed2i_mag.launch.py

Arguments
    camera_name   default: zed2i_mag
    camera_model  default: zed2i
    serial_number default: 0   (0 = auto-detect)
    use_sim_time  default: false
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def generate_launch_description() -> LaunchDescription:
    bringup = FindPackageShare('igvc_test_bringup')

    mag_only_config = PathJoinSubstitution(
        [bringup, 'config', 'zed2i_mag_only.yaml']
    )

    zed_camera_launch = os.path.join(
        get_package_share_directory('zed_wrapper'),
        'launch',
        'zed_camera.launch.py',
    )

    zed_node = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(zed_camera_launch),
        launch_arguments={
            'camera_name': LaunchConfiguration('camera_name'),
            'camera_model': LaunchConfiguration('camera_model'),
            'serial_number': LaunchConfiguration('serial_number'),
            'use_sim_time': LaunchConfiguration('use_sim_time'),
            # No TF/map TF in mag-only mode (positional tracking is disabled).
            'publish_tf': 'false',
            'publish_map_tf': 'false',
            'enable_gnss': 'false',
            'ros_params_override_path': mag_only_config,
        }.items(),
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            'camera_name',
            default_value='front_zed_camera_2i',
            description='Node/namespace name for the magnetometer-only ZED 2i.',
        ),
        DeclareLaunchArgument(
            'camera_model',
            default_value='zed2i',
            description='ZED camera model (zed2i).',
        ),
        DeclareLaunchArgument(
            'serial_number',
            default_value='0',
            description='ZED serial number (0 = auto-detect the connected camera).',
        ),
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='false',
            description='Use simulation clock if true.',
        ),
        zed_node,
    ])
