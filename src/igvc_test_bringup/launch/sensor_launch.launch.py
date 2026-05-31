import os
import re

import yaml

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, LogInfo, OpaqueFunction, TimerAction
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, TextSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


lidar_pkg = FindPackageShare('sllidar_ros2')
bringup = FindPackageShare('igvc_test_bringup')
gps_config = PathJoinSubstitution([bringup, 'config', 'zed_f9p.yaml'])




def _parse_array_param(raw_value: str):
    value = raw_value.replace('[', '').replace(']', '').replace(' ', '')
    items = value.split(',')
    if len(items) == 1 and items[0] == '':
        return []
    return items


def _safe_path_component(value: str) -> str:
    safe_value = re.sub(r'[^A-Za-z0-9_.-]+', '_', value).strip('._')
    return safe_value or 'camera'


def _is_front_camera(camera_name: str) -> bool:
    return camera_name.strip().lower() == 'front_zed_camera_x'


def _camera_params_path(
    base_params_path: str,
    camera_name: str,
    area_memory_path: str,
    enable_front_gnss: bool,
) -> str:
    with open(base_params_path, 'r', encoding='utf-8') as params_file:
        params = yaml.safe_load(params_file)

    ros_parameters = params.get('/**', {}).get('ros__parameters', {})
    pos_tracking = ros_parameters.get('pos_tracking', {})
    gnss_fusion = ros_parameters.get('gnss_fusion', {})

    # Keep GNSS fusion active only for the front camera when explicitly enabled.
    gnss_fusion['gnss_fusion_enabled'] = enable_front_gnss and _is_front_camera(camera_name)

    if pos_tracking.get('area_memory', False):
        os.makedirs(area_memory_path, exist_ok=True)
        area_memory_file = os.path.join(area_memory_path, f'{_safe_path_component(camera_name)}.area')
        pos_tracking['area_memory_db_path'] = area_memory_file
        pos_tracking['area_file_path'] = area_memory_file
        pos_tracking['save_area_memory_on_closing'] = True

    generated_params_dir = '/tmp/igvc_zed_params'
    os.makedirs(generated_params_dir, exist_ok=True)
    generated_params_path = os.path.join(
        generated_params_dir,
        f'{_safe_path_component(camera_name)}_common_stereo.yaml',
    )
    with open(generated_params_path, 'w', encoding='utf-8') as params_file:
        yaml.safe_dump(params, params_file, sort_keys=False)

    return generated_params_path


def launch_setup(context, *args, **kwargs):
    actions = []

    cam_names = _parse_array_param(LaunchConfiguration('cam_names').perform(context))
    cam_models = _parse_array_param(LaunchConfiguration('cam_models').perform(context))
    cam_serials = _parse_array_param(LaunchConfiguration('cam_serials').perform(context))
    cam_ids = _parse_array_param(LaunchConfiguration('cam_ids').perform(context))
    sim_ports = _parse_array_param(LaunchConfiguration('sim_ports').perform(context))

    namespace = LaunchConfiguration('namespace').perform(context)
    use_sim_time = LaunchConfiguration('use_sim_time').perform(context)
    sim_mode = LaunchConfiguration('sim_mode').perform(context)
    sim_address = LaunchConfiguration('sim_address').perform(context)
    disable_tf = LaunchConfiguration('disable_tf').perform(context).lower() == 'true'
    enable_front_gnss = LaunchConfiguration('enable_front_gnss').perform(context).lower() == 'true'
    ros_params_override_path = LaunchConfiguration('ros_params_override_path').perform(context)
    area_memory_path = LaunchConfiguration('area_memory_path').perform(context)

    # Resolve the override YAML path. Real hardware uses common_stereo_real.yaml;
    # Isaac Sim uses common_stereo_sim.yaml unless explicitly overridden.
    if not ros_params_override_path:
        default_config = 'common_stereo_sim.yaml' if sim_mode.lower() == 'true' else 'common_stereo_real.yaml'
        ros_params_override_path = os.path.join(
            get_package_share_directory('igvc_test_bringup'),
            'config',
            default_config,
        )

    if not os.path.isfile(ros_params_override_path):
        return [
            LogInfo(
                msg=TextSubstitution(
                    text=f'ZED ros_params_override_path not found: {ros_params_override_path}'
                )
            )
        ]

    actions.append(
        LogInfo(
            msg=TextSubstitution(
                text=f'ZED multi: applying ros_params_override_path={ros_params_override_path}'
            )
        )
    )

    num_cams = len(cam_names)

    if num_cams == 0:
        return [LogInfo(msg=TextSubstitution(text='No cameras configured in cam_names.'))]

    if num_cams != len(cam_models):
        return [
            LogInfo(
                msg=TextSubstitution(
                    text='`cam_models` must have the same length as `cam_names`.'
                )
            )
        ]

    if len(cam_serials) not in (0, num_cams):
        return [
            LogInfo(
                msg=TextSubstitution(
                    text='`cam_serials` must be empty or the same length as `cam_names`.'
                )
            )
        ]

    if len(cam_ids) not in (0, num_cams):
        return [
            LogInfo(
                msg=TextSubstitution(
                    text='`cam_ids` must be empty or the same length as `cam_names`.'
                )
            )
        ]

    if len(sim_ports) not in (0, num_cams):
        return [
            LogInfo(
                msg=TextSubstitution(
                    text='`sim_ports` must be empty or the same length as `cam_names`.'
                )
            )
        ]

    zed_camera_launch = os.path.join(
        get_package_share_directory('zed_wrapper'),
        'launch',
        'zed_camera.launch.py',
    )

    for idx, camera_name in enumerate(cam_names):
        camera_model = cam_models[idx]
        serial_number = cam_serials[idx] if len(cam_serials) == num_cams else '0'
        camera_id = cam_ids[idx] if len(cam_ids) == num_cams else '-1'
        sim_port = sim_ports[idx] if len(sim_ports) == num_cams else ''
        enable_gnss = 'true' if (enable_front_gnss and _is_front_camera(camera_name)) else 'false'

        publish_tf = 'false'
        publish_map_tf = 'false'
        if not disable_tf and idx == 0:
            publish_tf = 'true'
            publish_map_tf = 'true'

        info = (
            f'* Starting ZED camera: {camera_name} ({camera_model}), '
            f'publish_tf={publish_tf}, enable_gnss={enable_gnss}, use_sim_time={use_sim_time}'
        )
        actions.append(LogInfo(msg=TextSubstitution(text=info)))

        camera_params_path = _camera_params_path(
            ros_params_override_path,
            camera_name,
            area_memory_path,
            enable_front_gnss,
        )

        launch_arguments = {
            'camera_name': camera_name,
            'camera_model': camera_model,
            'serial_number': serial_number,
            'camera_id': camera_id,
            'use_sim_time': use_sim_time,
            'sim_mode': sim_mode,
            'enable_ipc': 'false',
            'publish_tf': publish_tf,
            'publish_map_tf': publish_map_tf,
            'enable_gnss': enable_gnss,
            'ros_params_override_path': camera_params_path,
        }

        if namespace:
            launch_arguments['namespace'] = namespace
        if sim_address:
            launch_arguments['sim_address'] = sim_address
        if sim_port:
            launch_arguments['sim_port'] = sim_port

        startup_delay_sec = float(idx) * 3.0
        actions.append(
            TimerAction(
                period=startup_delay_sec,
                actions=[
                    IncludeLaunchDescription(
                        PythonLaunchDescriptionSource(zed_camera_launch),
                        launch_arguments=launch_arguments.items(),
                    )
                ],
            )
        )

    return actions

    

def generate_launch_description():
    lidar_node = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([lidar_pkg, 'launch', 'sllidar_c1_launch.py'])
        ),
        launch_arguments={
            'frame_id': 'top_rplidar_c1_link',
        }.items(),
    )

    gps_node = Node(
        package='ublox_gps',
        executable='ublox_gps_node',
        name='ublox_gps_node',
        output='screen',
        parameters=[
            {'publish_odom': False},
            # {'publish_tf': True},
            gps_config,
        ],
    )
    l3gd20_node = Node(
        package='igvc_imu_interface',
        executable='l3gd20_heading_node',
        name='l3gd20_heading_node',
        output='screen',
        condition=IfCondition(LaunchConfiguration('enable_l3gd20')),
        parameters=[
            {
                'use_sim_time': ParameterValue(LaunchConfiguration('use_sim_time'), value_type=bool),
                'i2c_bus': ParameterValue(LaunchConfiguration('l3gd20_i2c_bus'), value_type=int),
                'i2c_address': ParameterValue(
                    LaunchConfiguration('l3gd20_i2c_address'),
                    value_type=int,
                ),
                'full_scale_dps': ParameterValue(
                    LaunchConfiguration('l3gd20_full_scale_dps'),
                    value_type=int,
                ),
                'sample_rate_hz': ParameterValue(
                    LaunchConfiguration('l3gd20_sample_rate_hz'),
                    value_type=float,
                ),
                'frame_id': LaunchConfiguration('l3gd20_frame_id'),
                'world_frame_id': LaunchConfiguration('l3gd20_world_frame_id'),
                'heading_topic': LaunchConfiguration('l3gd20_heading_topic'),
                'heading_degrees_topic': LaunchConfiguration('l3gd20_heading_degrees_topic'),
                'heading_quaternion_topic': LaunchConfiguration('l3gd20_heading_quaternion_topic'),
                'imu_topic': LaunchConfiguration('l3gd20_imu_topic'),
                'yaw_axis': LaunchConfiguration('l3gd20_yaw_axis'),
                'yaw_sign': ParameterValue(LaunchConfiguration('l3gd20_yaw_sign'), value_type=float),
                'initial_heading_rad': ParameterValue(
                    LaunchConfiguration('l3gd20_initial_heading_rad'),
                    value_type=float,
                ),
                'calibration_samples': ParameterValue(
                    LaunchConfiguration('l3gd20_calibration_samples'),
                    value_type=int,
                ),
                'deadband_dps': ParameterValue(
                    LaunchConfiguration('l3gd20_deadband_dps'),
                    value_type=float,
                ),
            }
        ],
    )
    return LaunchDescription(
        [
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
                default_value='[46941578,43593214,40636496]',
                description='Optional camera serials array (or empty).',
            ),
            DeclareLaunchArgument(
                'cam_ids',
                default_value='[]',
                description='Optional camera IDs array (or empty).',
            ),
            DeclareLaunchArgument(
                'sim_ports',
                default_value='[]',
                description='Optional simulation ports array (or empty).',
            ),
            DeclareLaunchArgument(
                'namespace',
                default_value='',
                description='Optional top-level namespace. Keep empty for existing IGVC topics.',
            ),
            DeclareLaunchArgument(
                'use_sim_time',
                default_value='false',
                description='Use simulation clock if true.',
            ),
            DeclareLaunchArgument(
                'sim_mode',
                default_value='false',
                description='Enable ZED simulation mode if true.',
            ),
            DeclareLaunchArgument(
                'sim_address',
                default_value='',
                description='Optional simulation server address. Leave empty to use the YAML default.',
            ),
            DeclareLaunchArgument(
                'disable_tf',
                default_value='true',
                description='Disable ZED odom/map TF publishing for all cameras.',
            ),
            DeclareLaunchArgument(
                'enable_front_gnss',
                default_value='false',
                description='Enable GNSS ingestion/fusion only for front_zed_camera_x when true.',
            ),
            DeclareLaunchArgument(
                'ros_params_override_path',
                default_value='',
                description='Path to a YAML file whose parameters override the ZED wrapper defaults. '
                            'If empty, common_stereo_real.yaml is used for real mode and '
                            'common_stereo_sim.yaml is used for sim mode.',
            ),
            DeclareLaunchArgument(
                'area_memory_path',
                default_value='/tmp/zed_area_memory',
                description='Directory for per-camera ZED area-memory files when area memory is enabled.',
            ),
            DeclareLaunchArgument(
                'enable_l3gd20',
                default_value='true',
                description='Start the L3GD20 I2C heading node.',
            ),
            DeclareLaunchArgument(
                'l3gd20_i2c_bus',
                default_value='0',
                description='Linux I2C bus for Jetson header pins 27 SDA / 28 SCL.',
            ),
            DeclareLaunchArgument(
                'l3gd20_i2c_address',
                default_value='0x6b',
                description='L3GD20 I2C address. Use 0x6a when SA0 is low.',
            ),
            DeclareLaunchArgument(
                'l3gd20_full_scale_dps',
                default_value='250',
                description='L3GD20 gyro range in degrees per second: 250, 500, or 2000.',
            ),
            DeclareLaunchArgument(
                'l3gd20_sample_rate_hz',
                default_value='95.0',
                description='Polling rate for the L3GD20 heading node.',
            ),
            DeclareLaunchArgument(
                'l3gd20_frame_id',
                default_value='l3gd20_link',
                description='Frame ID for the L3GD20 IMU message.',
            ),
            DeclareLaunchArgument(
                'l3gd20_world_frame_id',
                default_value='odom',
                description='World frame used by the heading quaternion topic.',
            ),
            DeclareLaunchArgument(
                'l3gd20_heading_topic',
                default_value='/navheading',
                description='Float64 heading topic in radians, wrapped to [-pi, pi].',
            ),
            DeclareLaunchArgument(
                'l3gd20_heading_degrees_topic',
                default_value='/navheading_deg',
                description='Float64 heading topic in degrees, wrapped to [0, 360).',
            ),
            DeclareLaunchArgument(
                'l3gd20_heading_quaternion_topic',
                default_value='/l3gd20/heading',
                description='QuaternionStamped heading topic in the configured world frame.',
            ),
            DeclareLaunchArgument(
                'l3gd20_imu_topic',
                default_value='/l3gd20/imu',
                description='Bias-corrected sensor_msgs/Imu output topic.',
            ),
            DeclareLaunchArgument(
                'l3gd20_yaw_axis',
                default_value='z',
                description='Gyro axis mounted as robot yaw: x, y, or z.',
            ),
            DeclareLaunchArgument(
                'l3gd20_yaw_sign',
                default_value='1.0',
                description='Sign correction for positive yaw direction.',
            ),
            DeclareLaunchArgument(
                'l3gd20_initial_heading_rad',
                default_value='0.0',
                description='Initial world-relative heading in radians.',
            ),
            DeclareLaunchArgument(
                'l3gd20_calibration_samples',
                default_value='250',
                description='Stationary samples used to calibrate gyro bias at startup.',
            ),
            DeclareLaunchArgument(
                'l3gd20_deadband_dps',
                default_value='0.03',
                description='Yaw-rate deadband in degrees per second after bias correction.',
            ),
            # OpaqueFunction(function=launch_setup),
            # lidar_node,
            gps_node,
            l3gd20_node,
        ]
    )
