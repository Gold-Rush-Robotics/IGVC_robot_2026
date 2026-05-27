"""Launch file for the IGVC task operator GUI."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    task_runner_node_arg = DeclareLaunchArgument(
        'task_runner_node',
        default_value='igvc_task_runner',
        description='Name of the running igvc_task_runner node.',
    )
    camera_topic_arg = DeclareLaunchArgument(
        'camera_topic',
        default_value='/front_zed_camera_x/rgb/image_raw',
        description='Camera image topic to display in the GUI.',
    )
    detection_topic_arg = DeclareLaunchArgument(
        'detection_topic',
        default_value='/detections',
        description='YOLO DetectionArray topic.',
    )

    gui_node = Node(
        package='igvc_task_gui',
        executable='task_gui',
        name='igvc_task_gui',
        output='screen',
        parameters=[{
            'task_runner_node': LaunchConfiguration('task_runner_node'),
            'camera_topic': LaunchConfiguration('camera_topic'),
            'detection_topic': LaunchConfiguration('detection_topic'),
        }],
    )

    return LaunchDescription([
        task_runner_node_arg,
        camera_topic_arg,
        detection_topic_arg,
        gui_node,
    ])
