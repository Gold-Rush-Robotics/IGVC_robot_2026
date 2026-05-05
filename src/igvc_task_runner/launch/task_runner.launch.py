"""Launch only the IGVC task runner node."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description() -> LaunchDescription:
    runner = FindPackageShare('igvc_task_runner')

    robot_mode_arg = DeclareLaunchArgument(
        'robot_mode', default_value='sim', choices=['sim', 'hardware'])
    task_mode_arg = DeclareLaunchArgument(
        'task_mode', default_value='selected', choices=['selected', 'auto'])
    selected_task_arg = DeclareLaunchArgument(
        'selected_task', default_value='full_course_2026')
    use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time', default_value='true', choices=['true', 'false'])

    task_runner = Node(
        package='igvc_task_runner',
        executable='task_runner',
        name='igvc_task_runner',
        output='screen',
        parameters=[
            PathJoinSubstitution([runner, 'config', 'task_runner.yaml']),
            {
                'use_sim_time': LaunchConfiguration('use_sim_time'),
                'robot_mode': LaunchConfiguration('robot_mode'),
                'task_mode': LaunchConfiguration('task_mode'),
                'selected_task': LaunchConfiguration('selected_task'),
            },
        ],
    )

    return LaunchDescription([
        robot_mode_arg,
        task_mode_arg,
        selected_task_arg,
        use_sim_time_arg,
        task_runner,
    ])
