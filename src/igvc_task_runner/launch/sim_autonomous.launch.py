"""Isaac/simulation autonomous launch with IGVC task runner."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def generate_launch_description() -> LaunchDescription:
    bringup = FindPackageShare('igvc_test_bringup')
    runner = FindPackageShare('igvc_task_runner')

    selected_task_arg = DeclareLaunchArgument(
        'selected_task', default_value='full_course_2026')
    task_mode_arg = DeclareLaunchArgument(
        'task_mode', default_value='selected', choices=['selected', 'auto'])
    use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time', default_value='true', choices=['true', 'false'])

    motor_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([bringup, 'launch', 'motor_controllers.launch.py'])),
        launch_arguments={
            'hardware_interface': 'IsaacDriveHardware',
            'use_sim_time': LaunchConfiguration('use_sim_time'),
        }.items(),
    )

    lane_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([bringup, 'launch', 'lane_follower.launch.py'])),
        launch_arguments={
            'gps_enabled': 'false',
            'force_identity_map_to_odom': 'true',
            'navigator_profile': 'fsd',
            'use_sim_time': LaunchConfiguration('use_sim_time'),
        }.items(),
    )

    runner_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([runner, 'launch', 'task_runner.launch.py'])),
        launch_arguments={
            'robot_mode': 'sim',
            'task_mode': LaunchConfiguration('task_mode'),
            'selected_task': LaunchConfiguration('selected_task'),
            'use_sim_time': LaunchConfiguration('use_sim_time'),
        }.items(),
    )

    return LaunchDescription([
        selected_task_arg,
        task_mode_arg,
        use_sim_time_arg,
        motor_launch,
        lane_launch,
        runner_launch,
    ])
