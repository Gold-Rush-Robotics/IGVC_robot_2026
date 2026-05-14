import os
import re

import yaml

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, LogInfo, OpaqueFunction, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, TextSubstitution


def _parse_array_param(raw_value: str):
    value = raw_value.replace('[', '').replace(']', '').replace(' ', '')
    items = value.split(',')
    if len(items) == 1 and items[0] == '':
        return []
    return items


def _safe_path_component(value: str) -> str:
    safe_value = re.sub(r'[^A-Za-z0-9_.-]+', '_', value).strip('._')
    return safe_value or 'camera'


def _camera_params_path(base_params_path: str, camera_name: str, area_memory_path: str) -> str:
    with open(base_params_path, 'r', encoding='utf-8') as params_file:
        params = yaml.safe_load(params_file)

    ros_parameters = params.get('/**', {}).get('ros__parameters', {})
    pos_tracking = ros_parameters.get('pos_tracking', {})

    if not pos_tracking.get('area_memory', False):
        return base_params_path

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

        publish_tf = 'false'
        publish_map_tf = 'false'
        if not disable_tf and idx == 0:
            publish_tf = 'true'
            publish_map_tf = 'true'

        info = (
            f'* Starting ZED camera: {camera_name} ({camera_model}), '
            f'publish_tf={publish_tf}, use_sim_time={use_sim_time}'
        )
        actions.append(LogInfo(msg=TextSubstitution(text=info)))

        camera_params_path = _camera_params_path(
            ros_params_override_path,
            camera_name,
            area_memory_path,
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
            'enable_gnss': 'true',
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
            OpaqueFunction(function=launch_setup),
        ]
    )
