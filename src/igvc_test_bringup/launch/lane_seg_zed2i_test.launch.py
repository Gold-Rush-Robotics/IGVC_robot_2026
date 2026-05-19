"""
lane_seg_zed2i_test.launch.py

Quick test launch — runs ONLY the lane segmentation node, pointed at the
/zed namespace (single ZED 2i) instead of the multi-camera
/front_zed_camera_x namespace used in the full pipeline.

Typical usage:
    export YOLOPV2_WEIGHTS=$PWD/models/yolopv2.pt
    ros2 launch igvc_test_bringup lane_seg_zed2i_test.launch.py

Arguments
    model_weights    absolute path   default: $YOLOPV2_WEIGHTS
    model_device     cpu | cuda:N    default: cuda:0
    model_half       true | false    default: true
    publish_overlay  true | false    default: true
    use_sim_time     true | false    default: false
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import (
    EnvironmentVariable,
    LaunchConfiguration,
    PathJoinSubstitution,
)
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description() -> LaunchDescription:

    # ── Arguments ──────────────────────────────────────────────────────
    model_weights_arg = DeclareLaunchArgument(
        'model_weights',
        default_value=EnvironmentVariable('YOLOPV2_WEIGHTS', default_value=''),
        description='Absolute path to yolopv2.pt TorchScript weights.')
    model_device_arg = DeclareLaunchArgument(
        'model_device', default_value='cuda:0',
        description='Torch device string (cpu, cuda:0, …).')
    model_half_arg = DeclareLaunchArgument(
        'model_half', default_value='true', choices=['true', 'false'])
    publish_overlay_arg = DeclareLaunchArgument(
        'publish_overlay', default_value='true', choices=['true', 'false'])
    use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time', default_value='false', choices=['true', 'false'])
    camera_height_m_arg = DeclareLaunchArgument(
        'camera_height_m', default_value='1.0',
        description='Static TF Z offset from base_link to camera optical frame.')
    camera_optical_frame_arg = DeclareLaunchArgument(
        'camera_optical_frame', default_value='zed_left_camera_frame_optical',
        description='Camera optical frame to connect to base_link in this quick test.')

    model_weights   = LaunchConfiguration('model_weights')
    model_device    = LaunchConfiguration('model_device')
    model_half      = LaunchConfiguration('model_half')
    publish_overlay = LaunchConfiguration('publish_overlay')
    use_sim_time    = LaunchConfiguration('use_sim_time')
    camera_height_m = LaunchConfiguration('camera_height_m')
    camera_optical_frame = LaunchConfiguration('camera_optical_frame')

    bringup = FindPackageShare('igvc_test_bringup')

    # ── Lane segmentation node (zed2i namespace override) ──────────────
    lane_segmentation_node = Node(
        package='igvc_lane_detection',
        executable='lane_segmentation_node',
        name='lane_segmentation_node',
        output='screen',
        parameters=[
            PathJoinSubstitution(
                [bringup, 'config', 'lane_segmentation_config.yaml']),
            {
                'use_sim_time':     use_sim_time,
                'model_weights':    model_weights,
                'model_device':     model_device,
                'model_half':       model_half,
                'publish_overlay':  publish_overlay,
                # ── /zed topic overrides ──────────────────────────
                'num_cameras': 1,
                'camera_topics':      ['/zed/zed_node/rgb/color/rect/image'],
                'depth_topics':       ['/zed/zed_node/depth/depth_registered'],
                'camera_info_topics': ['/zed/zed_node/rgb/color/rect/image/camera_info'],
                'odom_topic':         '/zed/zed_node/odom',
            },
        ],
    )

    # Bridge the ZED odometry into the robot TF tree so RViz can transform
    # /lane_costmap from base_link into odom during standalone testing.
    odom_tf_bridge_node = Node(
        package='igvc_lane_detection',
        executable='odom_tf_bridge_node',
        name='zed_odom_tf_bridge',
        output='screen',
        parameters=[{
            'odom_topic': '/zed/zed_node/odom',
            'odom_frame_id': 'odom',
            'base_frame_id': 'base_link',
            'publish_rate_hz': 100.0,
            'use_original_timestamp': False,
            'warn_odom_age_sec': 0.5,
            'max_odom_age_sec': 2.0,
        }],
    )

    # Connect camera optical frame to base_link for standalone testing.
    # This avoids pinhole fallback when a full robot TF tree is not running.
    static_base_to_camera_tf = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='static_base_to_camera_tf',
        output='screen',
        arguments=[
            '--x', '0.0',
            '--y', '0.0',
            '--z', camera_height_m,
            '--roll', '-1.57079632679',
            '--pitch', '0.0',
            '--yaw', '-1.57079632679',
            '--frame-id', 'base_link',
            '--child-frame-id', camera_optical_frame,
        ],
    )

    return LaunchDescription([
        model_weights_arg,
        model_device_arg,
        model_half_arg,
        publish_overlay_arg,
        use_sim_time_arg,
        camera_height_m_arg,
        camera_optical_frame_arg,
        odom_tf_bridge_node,
        static_base_to_camera_tf,
        lane_segmentation_node,
    ])
