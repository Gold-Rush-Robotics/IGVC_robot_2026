"""
mag_heading.launch.py

Launches the imu_filter_madgwick converter node, which fuses the ZED 2i's
raw IMU (gyro + accelerometer) with the magnetometer (micro-Tesla field) and
publishes a magnetic-north-referenced orientation -- i.e. a GPS/compass
heading -- as a sensor_msgs/Imu.

Inputs  (from the ZED 2i, see zed2i_mag.launch.py / zed2i_mag_only.yaml):
    /<camera_name>/zed_node/imu/data_raw   sensor_msgs/Imu          (gyro+accel)
    /<camera_name>/zed_node/imu/mag        sensor_msgs/MagneticField (Tesla)

Output:
    <output_topic>                         sensor_msgs/Imu          (orientation)
        The orientation quaternion's yaw is the heading. With
        world_frame=enu, yaw=0 points East and increases CCW (REP-103),
        referenced to magnetic north via the magnetometer.

Typical usage (bring up the camera separately, then this converter):
    ros2 launch igvc_test_bringup zed2i_mag.launch.py
    ros2 launch igvc_test_bringup mag_heading.launch.py

Or launch both together:
    ros2 launch igvc_test_bringup mag_heading.launch.py launch_camera:=true

Arguments
    camera_name    default: front_zed_camera_2i
    output_topic   default: /front_zed_camera_2i/imu/heading
    world_frame    enu | ned | nwu   default: enu
    use_sim_time   default: false
    launch_camera  default: false  (also start the ZED 2i mag bringup)
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

    camera_name = LaunchConfiguration('camera_name')
    output_topic = LaunchConfiguration('output_topic')
    world_frame = LaunchConfiguration('world_frame')
    use_sim_time = LaunchConfiguration('use_sim_time')

    # Optionally bring up the ZED 2i (mag + raw IMU) alongside the converter.
    zed_camera = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([bringup, 'launch', 'zed2i_mag.launch.py'])
        ),
        launch_arguments={
            'camera_name': camera_name,
            'use_sim_time': use_sim_time,
        }.items(),
        condition=IfCondition(LaunchConfiguration('launch_camera')),
    )

    # The converter: madgwick fuses raw IMU + magnetometer into an absolute
    # (magnetic-north) orientation. Its yaw is the heading.
    mag_heading_node = Node(
        package='imu_filter_madgwick',
        executable='imu_filter_madgwick_node',
        name='mag_heading_filter',
        output='screen',
        parameters=[{
            'use_sim_time': use_sim_time,
            'use_mag': True,            # fuse the magnetometer for absolute yaw
            'world_frame': world_frame,  # 'enu' -> REP-103 GPS heading convention
            'publish_tf': False,         # heading converter only; no TF
            'stateless': False,
            'remove_gravity_vector': False,
            'gain': 0.1,
            'mag_bias_x': 0.0,
            'mag_bias_y': 0.0,
            'mag_bias_z': 0.0,
        }],
        remappings=[
            ('imu/data_raw', ['/', camera_name, '/zed_node/imu/data_raw']),
            ('imu/mag',      ['/', camera_name, '/zed_node/imu/mag']),
            ('imu/data',     output_topic),
        ],
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            'camera_name',
            default_value='front_zed_camera_2i',
            description='Namespace of the ZED 2i publishing imu/data_raw and imu/mag.',
        ),
        DeclareLaunchArgument(
            'output_topic',
            default_value='/front_zed_camera_2i/imu/heading',
            description='Output sensor_msgs/Imu topic carrying the fused heading orientation.',
        ),
        DeclareLaunchArgument(
            'world_frame',
            default_value='enu',
            choices=['enu', 'ned', 'nwu'],
            description='Reference frame convention. enu = REP-103 GPS heading.',
        ),
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='false',
            choices=['true', 'false'],
            description='Use simulation clock if true.',
        ),
        DeclareLaunchArgument(
            'launch_camera',
            default_value='false',
            choices=['true', 'false'],
            description='Also start the ZED 2i mag bringup (zed2i_mag.launch.py).',
        ),
        zed_camera,
        mag_heading_node,
    ])
