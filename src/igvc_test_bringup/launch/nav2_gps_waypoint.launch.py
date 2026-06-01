"""
nav2_gps_waypoint.launch.py

Full bring-up for Nav2 GPS waypoint following on the IGVC robot, following the
nav2_gps_waypoint_follower_demo architecture.

Starts:
  1. (optional) motor controllers      — ros2_control + diff_drive  (hardware)
  2. (optional) sim_gps_spoofer         — fake /gps/fix + heading from ZED odom
  3. robot_localization                 — dual_ekf_navsat.launch.py (map->odom->base_link)
  4. Nav2 stack                         — navigation_no_docking.launch.py
  5. twist_stamper                      — cmd_vel_unstamped -> /diff_drive_controller/cmd_vel
  6. nav2_gps_waypoint_node             — sends the GPS goal to /follow_gps_waypoints

Usage
-----
  # Hardware, single waypoint:
  ros2 launch igvc_test_bringup nav2_gps_waypoint.launch.py \\
      goal_lat:=42.400510946 goal_lon:=-83.130518968

  # Simulation with spoofed GPS:
  ros2 launch igvc_test_bringup nav2_gps_waypoint.launch.py \\
      use_sim_time:=true launch_motors:=false use_sim_gps:=true \\
      goal_lat:=42.400510946 goal_lon:=-83.130518968

  # Follow a logged waypoints YAML instead of a single goal:
  ros2 launch igvc_test_bringup nav2_gps_waypoint.launch.py \\
      waypoints_file:=/path/to/waypoints.yaml

Arguments
---------
  use_sim_time        true | false                              (default: false)
  hardware_interface  CanInterface | IsaacDriveHardware         (default: CanInterface)
  launch_motors       Start ros2_control motor controllers      (default: true)
  use_sim_gps         Start sim_gps_spoofer (fake GPS/heading)  (default: false)
  nav2_params_file    Nav2 params YAML                          (default: nav2_gps_waypoint_config.yaml)
  rl_params_file      robot_localization params YAML            (default: dual_ekf_navsat_params.yaml)
  goal_lat            Target latitude                           (default: 42.400510946)
  goal_lon            Target longitude                          (default: -83.130518968)
  goal_yaw_deg        Target heading at the waypoint (deg)      (default: 0.0)
  waypoints_file      Optional YAML of multiple GPS waypoints   (default: "")
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description() -> LaunchDescription:
    bringup = FindPackageShare('igvc_test_bringup')

    # ── Arguments ─────────────────────────────────────────────────────────
    use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time', default_value='false',
        description='Use /clock from a simulator.')
    hardware_interface_arg = DeclareLaunchArgument(
        'hardware_interface', default_value='CanInterface',
        description='Hardware interface: CanInterface or IsaacDriveHardware.')
    launch_motors_arg = DeclareLaunchArgument(
        'launch_motors', default_value='true',
        description='Start the ros2_control motor controllers (hardware).')
    use_sim_gps_arg = DeclareLaunchArgument(
        'use_sim_gps', default_value='false',
        description='Start sim_gps_spoofer to fake /gps/fix and heading.')
    nav2_params_file_arg = DeclareLaunchArgument(
        'nav2_params_file',
        default_value=PathJoinSubstitution(
            [bringup, 'config', 'nav2_gps_waypoint_config.yaml']),
        description='Nav2 params YAML.')
    rl_params_file_arg = DeclareLaunchArgument(
        'rl_params_file',
        default_value=PathJoinSubstitution(
            [bringup, 'config', 'dual_ekf_navsat_params.yaml']),
        description='robot_localization params YAML.')
    goal_lat_arg = DeclareLaunchArgument(
        'goal_lat', default_value='42.400510946',
        description='Target waypoint latitude.')
    goal_lon_arg = DeclareLaunchArgument(
        'goal_lon', default_value='-83.130518968',
        description='Target waypoint longitude.')
    goal_yaw_deg_arg = DeclareLaunchArgument(
        'goal_yaw_deg', default_value='0.0',
        description='Target heading at the waypoint (degrees).')
    waypoints_file_arg = DeclareLaunchArgument(
        'waypoints_file', default_value='',
        description='Optional YAML file of multiple GPS waypoints.')

    use_sim_time = LaunchConfiguration('use_sim_time')
    hardware_interface = LaunchConfiguration('hardware_interface')
    launch_motors = LaunchConfiguration('launch_motors')
    use_sim_gps = LaunchConfiguration('use_sim_gps')
    nav2_params_file = LaunchConfiguration('nav2_params_file')
    rl_params_file = LaunchConfiguration('rl_params_file')
    goal_lat = LaunchConfiguration('goal_lat')
    goal_lon = LaunchConfiguration('goal_lon')
    goal_yaw_deg = LaunchConfiguration('goal_yaw_deg')
    waypoints_file = LaunchConfiguration('waypoints_file')

    # ── 1. Motor controllers (hardware only) ───────────────────────────────
    motor_controllers = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [bringup, 'launch', 'motor_controllers.launch.py'])),
        launch_arguments={
            'hardware_interface': hardware_interface,
            'use_sim_time': use_sim_time,
        }.items(),
        condition=IfCondition(launch_motors),
    )

    # ── 2. Simulated GPS + heading (sim only) ──────────────────────────────
    sim_gps_spoofer = Node(
        package='igvc_lane_detection',
        executable='sim_gps_spoofer_node',
        name='sim_gps_spoofer',
        output='screen',
        parameters=[{'use_sim_time': use_sim_time}],
        condition=IfCondition(use_sim_gps),
    )

    # ── 3. robot_localization (map -> odom -> base_link) ───────────────────
    localization = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [bringup, 'launch', 'dual_ekf_navsat.launch.py'])),
        launch_arguments={
            'use_sim_time': use_sim_time,
            'params_file': rl_params_file,
        }.items(),
    )

    # ── 4. Nav2 stack ──────────────────────────────────────────────────────
    nav2 = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [bringup, 'launch', 'navigation_no_docking.launch.py'])),
        launch_arguments={
            'use_sim_time': use_sim_time,
            'params_file': nav2_params_file,
            'autostart': 'true',
        }.items(),
    )

    # ── 5. Stamp Nav2's unstamped cmd_vel for the diff-drive controller ────
    twist_stamper = Node(
        package='twist_stamper',
        executable='twist_stamper',
        name='twist_stamper',
        output='screen',
        parameters=[{'use_sim_time': use_sim_time}, {'frame_id': 'base_link'}],
        remappings=[
            ('cmd_vel_in', '/diff_drive_controller/cmd_vel_unstamped'),
            ('cmd_vel_out', '/diff_drive_controller/cmd_vel'),
        ],
    )

    # ── 6. GPS waypoint follower client ────────────────────────────────────
    gps_waypoint_node = Node(
        package='igvc_lane_detection',
        executable='nav2_gps_waypoint_node',
        name='nav2_gps_waypoint',
        output='screen',
        parameters=[{
            'use_sim_time': use_sim_time,
            'goal_lat': goal_lat,
            'goal_lon': goal_lon,
            'goal_yaw_deg': goal_yaw_deg,
            'waypoints_file': waypoints_file,
            'localizer': 'robot_localization',
        }],
    )

    return LaunchDescription([
        use_sim_time_arg,
        hardware_interface_arg,
        launch_motors_arg,
        use_sim_gps_arg,
        nav2_params_file_arg,
        rl_params_file_arg,
        goal_lat_arg,
        goal_lon_arg,
        goal_yaw_deg_arg,
        waypoints_file_arg,
        motor_controllers,
        sim_gps_spoofer,
        localization,
        nav2,
        twist_stamper,
        gps_waypoint_node,
    ])
