import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, LogInfo
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, TextSubstitution


def generate_launch_description():
    """
    Launch a single ZED2i camera with custom object detection.
    
    This launch file is a test/example for running ZED2i with YOLOv12 IGVC 
    custom object detection enabled. The custom ONNX model path can be 
    specified via the 'custom_onnx_file' argument or overridden via 
    'ros_params_override_path'.
    """
    
    package_share_dir = get_package_share_directory('igvc_test_bringup')
    
    # Use the ZED2i custom detection config (OD-enabled) by default
    config_file = os.path.join(package_share_dir, 'config', 'zed2i_custom_detection.yaml')
    
    # Get ZED wrapper launch
    zed_camera_launch = os.path.join(
        get_package_share_directory('zed_wrapper'),
        'launch/include',
        'zed_camera.launch.py',
    )
    
    return LaunchDescription([
        DeclareLaunchArgument(
            'camera_name',
            default_value='zed2i',
            description='Name of the ZED2i camera node.',
        ),
        DeclareLaunchArgument(
            'camera_model',
            default_value='zed2i',
            description='ZED camera model (e.g., zed2i, zedx, zed2).',
        ),
        DeclareLaunchArgument(
            'serial_number',
            default_value='0',
            description='Serial number of the ZED camera. 0 for any available camera.',
        ),
        DeclareLaunchArgument(
            'camera_id',
            default_value='-1',
            description='Camera index if multiple cameras connected.',
        ),
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='false',
            description='Use simulated time.',
        ),
        DeclareLaunchArgument(
            'publish_tf',
            default_value='true',
            description='Publish TF transform from odom to camera.',
        ),
        DeclareLaunchArgument(
            'publish_map_tf',
            default_value='true',
            description='Publish TF transform from map to odom.',
        ),
        DeclareLaunchArgument(
            'custom_onnx_file',
            default_value='',
            description='Path to custom YOLO-like ONNX model file. If provided, overrides config file path.',
        ),
        DeclareLaunchArgument(
            'ros_params_override_path',
            default_value=config_file,
            description='Path to ROS parameters override YAML file for ZED config.',
        ),
        LogInfo(
            msg=TextSubstitution(
                text='Starting ZED2i camera with IGVC YOLOv12 custom object detection...'
            )
        ),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(zed_camera_launch),
            launch_arguments={
                'camera_name': LaunchConfiguration('camera_name'),
                'camera_model': LaunchConfiguration('camera_model'),
                'serial_number': LaunchConfiguration('serial_number'),
                'camera_id': LaunchConfiguration('camera_id'),
                'use_sim_time': LaunchConfiguration('use_sim_time'),
                'publish_tf': LaunchConfiguration('publish_tf'),
                'publish_map_tf': LaunchConfiguration('publish_map_tf'),
                'ros_params_override_path': LaunchConfiguration('ros_params_override_path'),
            }.items(),
        ),
    ])
