# IGVC_robot_2026

ROS 2 workspace for Gold Rush Robotics' 2026 IGVC robot. The repo contains robot description files, ros2_control hardware interfaces, bringup launch files, ZED/LiDAR/GPS integration, lane perception and navigation nodes, Isaac/Genesis simulation support, Docker runtime environments, and model training assets.

## Repository layout

| Path | Purpose |
| --- | --- |
| `src/` | ROS 2 packages for bringup, robot description, hardware, perception, simulation, and vendored drivers/interfaces. |
| `docker/` | Docker image definitions, including the Humble fused-drive runtime image. |
| `docker-compose.yml` | GPU-enabled compose services for the Jazzy ZED/dev environment and Humble fused-drive runtime. |
| `scripts/` | Workspace helper scripts for DDS setup and rosbag/data export utilities. |
| `training/` | YOLOv12 training workflow, dataset, training requirements, seed weights, and training outputs. |
| `IGVC_track_generator/` | Procedural IGVC-style course/track generation utilities. |
| `isaac/` | Isaac Sim project assets and extensions. |
| `isaacsim/` | Local Isaac Sim installation/support files. This is large and usually machine-specific. |
| `odrive_config/` | ODrive controller configuration snapshots. |
| `models/` | Runtime model weights such as `yolopv2.pt`; these are ignored by Git. |
| `build/`, `install/`, `log/` | Local colcon outputs. These are generated and ignored. |
| `yolopv2_bag/` | Local bag/data workspace for YOLOPv2 experimentation. |

## Runtime targets

The workspace is used in two main ROS environments:

| Target | Use case | Notes |
| --- | --- | --- |
| ROS 2 Jazzy on host | Local development, debug GUI, ZED dev container, and tools. | Host tools can see Docker topics when the shared DDS helper is sourced. |
| ROS 2 Humble in Docker | Fused-drive runtime for compatibility with Humble-only dependencies. | Uses `ros:humble-ros-base` through `docker/Dockerfile.humble-fused-drive`. |

## ROS packages

### Project packages

| Package | Role |
| --- | --- |
| `igvc_test_bringup` | Launch/config package for motor controllers, fused drive, lane segmentation, ZED odometry, Nav2, RViz, teleop, and simulation bringup. |
| `igvc_test_description` | Robot URDF/Xacro, meshes, sensor mounts, ros2_control tags, and generated URDFs. |
| `grr_hardware` | ros2_control hardware plugins for robot drive interfaces. Exports `grr_hardware/CanInterface` and `grr_hardware/IsaacDriveHardware`. |
| `igvc_lane_detection` | Python perception/navigation stack: lane detection, YOLOPv2 lane segmentation, localization, navigator, odom/TF bridge, and projection utilities. |
| `igvc_simulation_interface` | Python client wrapper around simulation interfaces for loading worlds, spawning entities, stepping/resetting simulation, and Isaac/Genesis integration. |
| `igvc_lidar_test` | LiDAR test package that depends on `sllidar_ros2`. |
| `debug_gui` | PyQt debug GUI for direct robot commands and camera/image feedback. |

### Vendored or third-party packages

| Package group | Role |
| --- | --- |
| `sllidar_ros2` | Slamtec/RPLiDAR ROS 2 driver and view launch files. |
| `zed_description` | Stereolabs ZED camera description package. |
| `ros_odrive/` | ODrive CAN and ros2_control packages plus a botwheel explorer demo. |
| `ros2-ublox-zedf9p/` | u-blox ZED-F9P GPS driver, messages, serialization, and metapackage. |
| `genesis_ros/` | Genesis simulation packages, simulation interfaces, and ros2_control demos. |

## Key launch files

Launch files live mainly in `src/igvc_test_bringup/launch/`.

| Launch file | Purpose |
| --- | --- |
| `igvc_fused_drive.launch.py` | Main fused-drive stack. Brings up motor controllers, lane segmentation/Nav2, odom-to-TF, and twist stamping. |
| `motor_controllers.launch.py` | Robot description, `ros2_control_node`, joint state broadcaster, and diff drive controller. |
| `teleop.launch.py` | Includes motor controllers and adds joystick teleop via `joy` and `teleop_twist_joy`. |
| `lane_segmentation.launch.py` | YOLOPv2 lane segmentation, localization, navigator, and Nav2. |
| `lane_follower.launch.py` | Classic lane follower path using the non-deep-learning lane detector. |
| `navigation_no_docking.launch.py` | Nav2 bringup without docking behavior. |
| `zed_multi.launch.py` | Multi-ZED camera bringup. |
| `zed_multi_fused_odom.launch.py` | ZED odometry plus robot localization fusion. |
| `simulation_launch.launch.yaml` | Simulation bringup used by the ZED/dev compose service. |
| `simulation_interface.launch.yaml` | Simulation interface launch entry. |
| `rviz.launch.py` | RViz startup using the project config. |

Common commands:

```bash
ros2 launch igvc_test_bringup motor_controllers.launch.py hardware_interface:=CanInterface
ros2 launch igvc_test_bringup teleop.launch.py hardware_interface:=CanInterface
ros2 launch igvc_test_bringup lane_segmentation.launch.py model_weights:=$PWD/models/yolopv2.pt
ros2 launch igvc_test_bringup igvc_fused_drive.launch.py hardware_interface:=IsaacDriveHardware use_sim_time:=true
```

## Build and local workspace use

From the repository root:

```bash
source /opt/ros/jazzy/setup.bash
colcon build --symlink-install --base-paths src
source install/setup.bash
```

For a narrower runtime build matching the Humble container package set:

```bash
colcon build --symlink-install \
	--base-paths src \
	--packages-select \
		zed_description \
		igvc_test_description \
		grr_hardware \
		igvc_lane_detection \
		igvc_test_bringup
```

Useful local tools:

```bash
ros2 run debug_gui debugger
ros2 run igvc_lane_detection lane_segmentation_node
ros2 run igvc_lane_detection navigation_node
ros2 run igvc_simulation_interface simulation_interface
```

## Humble Docker bringup

The repo includes a ROS 2 Humble container for running the fused-drive stack on systems where the host ROS install is different, such as Jazzy.

Build the dependency image once:

```bash
docker compose build igvc_humble_fused_drive
```

Launch the Humble fused-drive stack:

```bash
docker compose up igvc_humble_fused_drive
```

The container runs:

```bash
ros2 launch igvc_test_bringup igvc_fused_drive.launch.py hardware_interface:=IsaacDriveHardware use_sim_time:=true
```

The Humble image definition lives in `docker/Dockerfile.humble-fused-drive`. The compose service keeps `build`, `install`, and `log` in Docker volumes so restarts can reuse colcon output instead of rebuilding from scratch each time.

To force a clean Humble workspace rebuild:

```bash
docker compose down
docker volume rm igvc_robot_2026_igvc_humble_build igvc_robot_2026_igvc_humble_install igvc_robot_2026_igvc_humble_log
docker compose up igvc_humble_fused_drive
```

## Docker services

| Service | Image | Purpose |
| --- | --- | --- |
| `igvc_dev_zed` | `ghcr.io/gold-rush-robotics/dev-zed:5.1.0-13.0.0` | Jazzy/ZED development container. Builds `igvc_test_bringup` and launches `simulation_launch.launch.yaml`. |
| `igvc_humble_fused_drive` | Built from `docker/Dockerfile.humble-fused-drive` | Humble runtime container for `igvc_fused_drive.launch.py` with GPU, host networking, DDS profile, and persistent colcon volumes. |

Both services mount the repository at `/root/ros2_ws/src/IGVC_robot_2026`, use host networking, expose `/dev`, share `/tmp/.X11-unix`, and request NVIDIA GPU access.

## Shared DDS profile

Humble-in-Docker and local Jazzy tools must use the same DDS settings to see the same ROS graph. Both compose services use the shared Fast DDS profile at:

```text
src/igvc_test_bringup/config/fastdds_udp.xml
```

Before running local Jazzy commands that need to see Docker topics, source:

```bash
source scripts/use_igvc_dds.sh
source /opt/ros/jazzy/setup.bash
source install/setup.bash
ros2 topic list
```

The helper sets `RMW_IMPLEMENTATION=rmw_fastrtps_cpp`, `ROS_DOMAIN_ID`, `ROS_LOCALHOST_ONLY=0`, `FASTDDS_BUILTIN_TRANSPORTS=UDPv4`, and the shared Fast DDS profile path. If you change `ROS_DOMAIN_ID`, use the same value for both the local shell and Docker compose.

## Perception and navigation

The main perception/navigation stack is in `igvc_lane_detection` and is launched through `igvc_test_bringup`.

| Node | Executable | Purpose |
| --- | --- | --- |
| Lane detector | `lane_detection_node` | Classic image/depth lane detection and occupancy grid generation. |
| YOLOPv2 segmenter | `lane_segmentation_node` | Deep segmentation pipeline using `yolopv2.pt`, depth projection, occupancy grid output, overlays, and markers. |
| Localization | `localization_node` | Localization state handling for sim, dead reckoning, and GPS-enabled modes. |
| Navigator | `navigation_node` | Converts lane/costmap information into Nav2 waypoints/goals. |
| Odom/TF bridge | `odom_tf_bridge_node` | Publishes transform data from odometry when needed. |
| Multi-camera detector | `multi_camera_lane_detection_node` | Multi-camera lane fusion path. |

Runtime model weights are expected at `models/yolopv2.pt` or can be passed through `model_weights:=...`. The Humble Docker service sets:

```text
YOLOPV2_WEIGHTS=/root/ros2_ws/src/IGVC_robot_2026/models/yolopv2.pt
```

To fetch or prepare YOLOPv2 runtime weights, see:

```bash
src/igvc_lane_detection/scripts/fetch_yolopv2_weights.sh
```

## Robot description and control

Robot description assets live in `src/igvc_test_description/`.

| Path | Purpose |
| --- | --- |
| `urdf/robots/test_robot.urdf.xacro` | Main robot Xacro entry point. |
| `urdf/control/ros2_control_info.urdf.xacro` | ros2_control hardware interface wiring. |
| `urdf/parts/` | Body, ZED camera, GPS antenna, and RPLiDAR Xacro parts. |
| `meshes/` | Robot and sensor meshes. |
| `igvc_robot.urdf`, `test_robot.urdf` | Generated or exported URDF artifacts. |

`grr_hardware` exports ros2_control system plugins:

| Plugin | Use case |
| --- | --- |
| `grr_hardware/CanInterface` | Physical robot CAN/motor-controller interface. |
| `grr_hardware/IsaacDriveHardware` | Isaac/simulation-facing drive interface. |

Motor controller parameters are in `src/igvc_test_bringup/config/controllers.yaml`.

## Sensors and configuration

Important bringup configs live in `src/igvc_test_bringup/config/`.

| Config | Purpose |
| --- | --- |
| `controllers.yaml` | ros2_control controller configuration. |
| `xbox-holonomic.config.yaml` | Joystick teleop mapping. |
| `lane_segmentation_config.yaml` | YOLOPv2 lane segmentation parameters. |
| `lane_detection_config.yaml` | Classic lane detector parameters. |
| `multi_camera_lane_detection.yaml` | Multi-camera lane detection parameters. |
| `navigator_config.yaml` | Lane navigator and waypoint behavior. |
| `nav2_lane_follow_config.yaml` | Nav2 parameters for lane-following behavior. |
| `igvc_nav_to_pose_bt.xml` | Nav2 behavior tree XML. |
| `zed_multi_ekf.yaml` | Robot localization EKF for multi-ZED odometry. |
| `zed_f9p.yaml` | u-blox ZED-F9P GPS parameters. |
| `common_stereo.yaml` | Shared ZED stereo camera configuration copied into the ZED wrapper container. |
| `twist_mux.yaml` | Twist multiplexer configuration. |
| `config.rviz` | RViz visualization config. |
| `fastdds_udp.xml` | Shared Fast DDS UDP profile. |

## Training assets

YOLOv12 training files live under `training/` so the ROS workspace root stays focused on bringup and runtime code.

```bash
cd training
pip install -r training_requirements.txt
./train_yolov12.sh --help
```

See `training/README.md` for the full training workflow. Training data is in `training/dataset/`, seed weights are in `training/weights/`, and outputs are written under `training/runs/`.

Training layout:

| Path | Purpose |
| --- | --- |
| `training/train_yolov12.py` | Python training entry point using Ultralytics. |
| `training/train_yolov12.sh` | Shell wrapper with common defaults. |
| `training/training_requirements.txt` | Python training dependencies. |
| `training/dataset/` | Roboflow/YOLO dataset. |
| `training/weights/` | Seed model weights such as `yolov12n.pt`. |
| `training/runs/` | Training outputs. |

## Simulation and course generation

Simulation support is split across several directories:

| Path | Purpose |
| --- | --- |
| `src/igvc_simulation_interface/` | ROS 2 Python interface for simulation services. |
| `src/genesis_ros/` | Genesis simulator bridge packages and interface definitions. |
| `isaac/` | Isaac-specific assets/extensions. |
| `isaacsim/` | Local Isaac Sim installation tree and helper scripts. |
| `IGVC_track_generator/` | Procedural IGVC course generator; see `IGVC_track_generator/README.md`. |

The track generator can produce IGVC-style course imagery and related assets for simulation/training experiments.

## Data utilities

Workspace-level scripts in `scripts/`:

| Script | Purpose |
| --- | --- |
| `use_igvc_dds.sh` | Exports shared DDS settings for local shells. |
| `bag_to_csv_and_images.py` | Converts ROS bag data into CSV/images for inspection or dataset preparation. |
| `bag_to_yolopv2_overlay.py` | Generates YOLOPv2 overlay artifacts from bag data. |

## Generated and ignored content

These directories/files are treated as generated, local, or large artifacts:

| Pattern | Reason |
| --- | --- |
| `build/`, `install/`, `log/` | Colcon outputs. |
| `training/dataset/`, `training/runs/`, `training/weights/*.pt` | Training data, outputs, and local weights. |
| `models/`, `*.pt` | Runtime model weights. |
| `isaacsim/` | Local Isaac Sim install/support tree. |
| `rosbags/`, `rosbag_export_test/`, `yolopv2_bag/` | Local bag/export data. |
| `**.pyc`, `.venv/` | Python generated files and virtual environments. |

## Common workflows

### Run the Humble fused-drive stack

```bash
docker compose build igvc_humble_fused_drive
docker compose up igvc_humble_fused_drive
```

### Inspect Humble topics from a Jazzy shell

```bash
source scripts/use_igvc_dds.sh
source /opt/ros/jazzy/setup.bash
source install/setup.bash
ros2 topic list
```

### Run teleop with physical CAN hardware

```bash
source /opt/ros/jazzy/setup.bash
source install/setup.bash
ros2 launch igvc_test_bringup teleop.launch.py hardware_interface:=CanInterface
```

### Run teleop against the Isaac drive interface

```bash
source /opt/ros/jazzy/setup.bash
source install/setup.bash
ros2 launch igvc_test_bringup teleop.launch.py hardware_interface:=IsaacDriveHardware use_sim_time:=true
```

### Run the debug GUI

```bash
source /opt/ros/jazzy/setup.bash
source install/setup.bash
ros2 run debug_gui debugger
```

### Validate compose and training script syntax

```bash
bash -n training/train_yolov12.sh
python3 -m py_compile training/train_yolov12.py
docker compose -f docker-compose.yml config >/tmp/igvc_compose_config.yml
```

## Notes for maintainers

- Keep ROS launch/config files inside `src/igvc_test_bringup/` unless they belong to a specific package.
- Keep Docker image definitions under `docker/` and wire them from `docker-compose.yml`.
- Keep training-only data and scripts under `training/`.
- Keep runtime model weights in `models/` or pass absolute paths with launch arguments.
- Use the shared DDS profile whenever local tools and Docker containers need to participate in the same ROS graph.
