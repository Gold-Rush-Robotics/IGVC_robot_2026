import time
import rclpy
import os
import tempfile

from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
from ament_index_python.packages import get_package_share_directory
from std_msgs.msg import String

from simulation_interfaces.msg import Result, SimulatorFeatures, SimulationState, WorldResource, Resource
from simulation_interfaces.srv import (
    GetSimulatorFeatures, LoadWorld, SpawnEntity, DeleteEntity,
    GetEntityState, SetEntityState, SetSimulationState,
    StepSimulation, ResetSimulation, UnloadWorld
)

from geometry_msgs.msg import PoseStamped, Quaternion, Point

# Container path root (detected from workspace structure)
CONTAINER_DEVENV_ROOT = '/workspace/DevEnv'

# Service names
GET_FEATURES_SERVICE = "/get_simulator_features"
LOAD_WORLD_SERVICE = "/load_world"
SPAWN_ENTITY_SERVICE = "/spawn_entity"
DELETE_ENTITY_SERVICE = "/delete_entity"
GET_ENTITY_STATE_SERVICE = "/get_entity_state"
SET_ENTITY_STATE_SERVICE = "/set_entity_state"
SET_SIMULATION_STATE_SERVICE = "/set_simulation_state"
STEP_SIMULATION_SERVICE = "/step_simulation"
RESET_SIMULATION_SERVICE = "/reset_simulation"
UNLOAD_WORLD_SERVICE = "/unload_world"


class SimulationInterface(Node):
    """ROS2 node for interfacing with Isaac Sim simulation."""

    def __init__(self):
        super().__init__('simulation_interface_node')
        self.get_logger().info("Simulation Interface Node has been started.")

        # Create service clients
        self.get_features_client = self.create_client(GetSimulatorFeatures, GET_FEATURES_SERVICE)
        self.load_world_client = self.create_client(LoadWorld, LOAD_WORLD_SERVICE)
        self.spawn_entity_client = self.create_client(SpawnEntity, SPAWN_ENTITY_SERVICE)
        self.delete_entity_client = self.create_client(DeleteEntity, DELETE_ENTITY_SERVICE)
        self.get_entity_state_client = self.create_client(GetEntityState, GET_ENTITY_STATE_SERVICE)
        self.set_entity_state_client = self.create_client(SetEntityState, SET_ENTITY_STATE_SERVICE)
        self.set_simulation_state_client = self.create_client(SetSimulationState, SET_SIMULATION_STATE_SERVICE)
        self.step_simulation_client = self.create_client(StepSimulation, STEP_SIMULATION_SERVICE)
        self.reset_simulation_client = self.create_client(ResetSimulation, RESET_SIMULATION_SERVICE)
        self.unload_world_client = self.create_client(UnloadWorld, UNLOAD_WORLD_SERVICE)

        # Subscribe to robot_description with transient_local QoS to receive latched messages
        robot_description_qos = QoSProfile(
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE
        )
        self.robot_description_subscriber = self.create_subscription(
            String, '/robot_description', self.robot_description_callback, robot_description_qos
        )
        self.robot_description = ""
        self._warned_host_path = False

    def robot_description_callback(self, msg):
        """Callback for robot_description topic."""
        self.get_logger().info("Received robot description update.")
        self.robot_description = msg.data
        if not self.robot_description:
            self.get_logger().warn("Robot description is empty!")

    def _map_to_host_path(self, container_path: str) -> str:
        """Map a container path to host path using DEVENV_HOST_PATH.
        
        If DEVENV_HOST_PATH is not set, warns once and returns the original path.
        """
        host_root = os.environ.get('DEVENV_HOST_PATH')
        
        if not host_root:
            if not self._warned_host_path:
                self.get_logger().warn(
                    'DEVENV_HOST_PATH environment variable is not set. '
                    'Isaac Sim may not be able to access container paths. '
                    'Set DEVENV_HOST_PATH to your host DevEnv directory.'
                )
                self._warned_host_path = True
            return container_path
        
        # Check if path starts with container root
        abs_path = os.path.abspath(container_path)
        if abs_path.startswith(CONTAINER_DEVENV_ROOT):
            # Replace container root with host root
            rel_path = os.path.relpath(abs_path, CONTAINER_DEVENV_ROOT)
            host_path = os.path.join(host_root, rel_path)
            self.get_logger().debug(f'Mapped {container_path} -> {host_path}')
            return host_path
        
        return container_path

    # ************************************
    # Simulation service methods
    # ************************************

    def get_features(self) -> SimulatorFeatures:
        """Get simulator features."""
        if not self.get_features_client.wait_for_service(timeout_sec=5.0):
            self.get_logger().error('GetSimulatorFeatures service not available')
            return None

        req = GetSimulatorFeatures.Request()
        future = self.get_features_client.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        if future.result():
            return future.result().features
        else:
            self.get_logger().error("Failed to call get_simulator_features service")
            return None

    def load_world(self, uri: str) -> bool:
        """Load a world/scene from URI."""
        if not self.load_world_client.wait_for_service(timeout_sec=5.0):
            self.get_logger().error('LoadWorld service not available')
            return False

        req = LoadWorld.Request()
        req.world_resource.uri = uri
        future = self.load_world_client.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        if future.result() and future.result().result.result == Result.RESULT_OK:
            self.get_logger().info(f"Loaded world: {uri}")
            return True
        else:
            error_msg = future.result().result.error_message if future.result() else "no response"
            self.get_logger().error(f"Failed to load world {uri}: {error_msg}")
            return False

    def spawn_entity(self, uri: str, name: str, initial_pose: PoseStamped) -> bool:
        """Spawn an entity from URI."""
        if not self.spawn_entity_client.wait_for_service(timeout_sec=5.0):
            self.get_logger().error('SpawnEntity service not available')
            return False

        req = SpawnEntity.Request()
        req.name = name
        req.entity_resource.uri = uri
        req.allow_renaming = True
        req.initial_pose = initial_pose
        future = self.spawn_entity_client.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        if future.result() and future.result().result.result == Result.RESULT_OK:
            self.get_logger().info(f"Spawned entity: {name}")
            return True
        else:
            error_msg = future.result().result.error_message if future.result() else "no response"
            self.get_logger().error(f"Failed to spawn {name}: {error_msg}")
            return False

    def delete_entity(self, name: str) -> bool:
        """Delete an entity by name."""
        if not self.delete_entity_client.wait_for_service(timeout_sec=5.0):
            self.get_logger().error('DeleteEntity service not available')
            return False

        req = DeleteEntity.Request()
        req.entity = name
        future = self.delete_entity_client.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        if future.result() and future.result().result.result == Result.RESULT_OK:
            self.get_logger().info(f"Deleted entity: {name}")
            return True
        else:
            error_msg = future.result().result.error_message if future.result() else "no response"
            self.get_logger().error(f"Failed to delete entity '{name}': {error_msg}")
            return False

    def get_entity_state(self, name: str) -> PoseStamped:
        """Get the state/pose of an entity."""
        if not self.get_entity_state_client.wait_for_service(timeout_sec=5.0):
            self.get_logger().error('GetEntityState service not available')
            return None

        req = GetEntityState.Request()
        req.entity = name
        future = self.get_entity_state_client.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        if future.result() and future.result().result.result == Result.RESULT_OK:
            return future.result().state
        else:
            error_msg = future.result().result.error_message if future.result() else "no response"
            self.get_logger().error(f"Failed to get entity state for '{name}': {error_msg}")
            return None

    def set_entity_state(self, name: str, pose: PoseStamped) -> bool:
        """Set the state/pose of an entity."""
        if not self.set_entity_state_client.wait_for_service(timeout_sec=5.0):
            self.get_logger().error('SetEntityState service not available')
            return False

        req = SetEntityState.Request()
        req.entity = name
        req.state.pose = pose.pose
        future = self.set_entity_state_client.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        if future.result() and future.result().result.result == Result.RESULT_OK:
            return True
        else:
            error_msg = future.result().result.error_message if future.result() else "no response"
            self.get_logger().error(f"Failed to set entity state for {name}: {error_msg}")
            return False

    def set_simulation_state(self, state: int) -> bool:
        """Set simulation state (playing, paused, stopped)."""
        if not self.set_simulation_state_client.wait_for_service(timeout_sec=5.0):
            self.get_logger().error('SetSimulationState service not available')
            return False

        req = SetSimulationState.Request()
        req.state.state = state
        future = self.set_simulation_state_client.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        if future.result() and future.result().result.result == Result.RESULT_OK:
            return True
        else:
            error_msg = future.result().result.error_message if future.result() else "no response"
            self.get_logger().error(f"Failed to set simulation state: {error_msg}")
            return False

    def step_simulation(self, steps: int = 1) -> bool:
        """Step simulation by N steps."""
        if not self.step_simulation_client.wait_for_service(timeout_sec=5.0):
            self.get_logger().error('StepSimulation service not available')
            return False

        req = StepSimulation.Request()
        req.steps = steps
        future = self.step_simulation_client.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        if future.result() and future.result().result.result == Result.RESULT_OK:
            return True
        else:
            error_msg = future.result().result.error_message if future.result() else "no response"
            self.get_logger().error(f"Failed to step simulation: {error_msg}")
            return False

    def reset_simulation(self) -> bool:
        """Reset simulation state."""
        if not self.reset_simulation_client.wait_for_service(timeout_sec=5.0):
            self.get_logger().error('ResetSimulation service not available')
            return False

        req = ResetSimulation.Request()
        req.scope = ResetSimulation.Request.SCOPE_STATE
        future = self.reset_simulation_client.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        if future.result() and future.result().result.result == Result.RESULT_OK:
            return True
        else:
            error_msg = future.result().result.error_message if future.result() else "no response"
            self.get_logger().error(f"Failed to reset simulation: {error_msg}")
            return False

    def unload_world(self) -> bool:
        """Unload the current world."""
        if not self.unload_world_client.wait_for_service(timeout_sec=5.0):
            self.get_logger().error('UnloadWorld service not available')
            return False

        req = UnloadWorld.Request()
        future = self.unload_world_client.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        if future.result() and future.result().result.result == Result.RESULT_OK:
            return True
        else:
            error_msg = future.result().result.error_message if future.result() else "no response"
            self.get_logger().error(f"Failed to unload world: {error_msg}")
            return False

    # ************************************
    # High-level convenience methods
    # ************************************

    def load_field_usd(self, field_usd_path: str = None) -> bool:
        """Load field1.usd from isaac/assets or a custom path.

        Args:
            field_usd_path: Optional custom path to field USD. If None, tries
                           to locate isaac/assets/field1.usd in workspace.
        Returns:
            True if world loaded successfully, False otherwise.
        """
        if field_usd_path is None:
            # Try to locate field1.usd in workspace
            try:
                pkg_share = get_package_share_directory('igvc_simulation_interface')
                workspace_root = os.path.normpath(os.path.join(pkg_share, '..', '..', '..'))
                field_usd_path = os.path.join(workspace_root, 'isaac', 'assets', 'field1.usd')
            except Exception:
                field_usd_path = None

            if field_usd_path is None or not os.path.exists(field_usd_path):
                # Try current working directory as fallback
                alt = os.path.join(os.getcwd(), 'isaac', 'assets', 'field2.usd')
                if os.path.exists(alt):
                    field_usd_path = alt

        if field_usd_path is None or not os.path.exists(field_usd_path):
            self.get_logger().error(f'Field USD not found at {field_usd_path}')
            return False

        # Map to host path for Isaac Sim access
        host_path = self._map_to_host_path(field_usd_path)
        self.get_logger().info(f'Loading field USD from {host_path}')
        return self.load_world(f"file://{host_path}")

    def spawn_robot_from_description(self, entity_name: str = 'igvc_robot',
                                      initial_pose: PoseStamped = None,
                                      timeout: float = 5.0) -> bool:
        """Spawn robot URDF from /robot_description topic.

        Waits for robot_description if not already received, writes URDF to
        temp file, then spawns entity.

        Args:
            entity_name: Name for the spawned robot entity.
            initial_pose: Optional initial pose. Defaults to origin.
            timeout: Seconds to wait for robot_description.
        Returns:
            True if robot spawned successfully, False otherwise.
        """
        # Wait for robot_description if not already received
        if not self.robot_description:
            self.get_logger().info('Waiting for robot_description...')
            t0 = time.time()
            while rclpy.ok() and (time.time() - t0) < timeout:
                rclpy.spin_once(self, timeout_sec=0.1)
                if self.robot_description:
                    break

        if not self.robot_description:
            self.get_logger().error('No robot_description received')
            return False

        # Write URDF to a location accessible by both container and host
        # Use workspace .tmp dir so path mapping works
        try:
            pkg_share = get_package_share_directory('igvc_simulation_interface')
            workspace_root = os.path.normpath(os.path.join(pkg_share, '..', '..', '..'))
        except Exception:
            workspace_root = os.getcwd()
        
        tmpdir = os.path.join(workspace_root, '.tmp')
        os.makedirs(tmpdir, exist_ok=True)
        urdf_path = os.path.join(tmpdir, f'robot_{entity_name}.urdf')
        with open(urdf_path, 'w') as f:
            f.write(self.robot_description)
        self.get_logger().info(f'Wrote robot URDF to {urdf_path}')

        # Set default pose at origin if not provided
        if initial_pose is None:
            initial_pose = PoseStamped()
            initial_pose.pose.position = Point(x=0.0, y=0.0, z=0.0)
            initial_pose.pose.orientation = Quaternion(w=1.0, x=0.0, y=0.0, z=0.0)

        # Map to host path for Isaac Sim access
        host_path = self._map_to_host_path(urdf_path)
        return self.spawn_entity(f"file://{host_path}", entity_name, initial_pose)

    def spawn_robot_from_usd(self,
                             entity_name: str = 'igvc_robot',
                             initial_pose: PoseStamped = None,
                             robot_usd_path: str = None) -> bool:
        """Spawn robot USD from a file path.

        Args:
            entity_name: Name for the spawned robot entity.
            initial_pose: Optional initial pose. Defaults to origin.
            robot_usd_path: Optional custom USD path. If None, uses
                            isaac/assets/test_igvc_drivebase..usd.
        Returns:
            True if robot spawned successfully, False otherwise.
        """
        if robot_usd_path is None:
            robot_usd_path = os.path.join(
                CONTAINER_DEVENV_ROOT,
                'jazzy_ws',
                'IGVC_robot_2026',
                'isaac',
                'assets',
                'test_igvc_drivebase..usd'
            )

        if not os.path.exists(robot_usd_path):
            self.get_logger().error(f'Robot USD not found at {robot_usd_path}')
            return False

        # Set default pose at origin if not provided
        if initial_pose is None:
            initial_pose = PoseStamped()
            initial_pose.pose.position = Point(x=0.0, y=0.0, z=0.0)
            initial_pose.pose.orientation = Quaternion(w=1.0, x=0.0, y=0.0, z=0.0)

        # Map to host path for Isaac Sim access
        host_path = self._map_to_host_path(robot_usd_path)
        self.get_logger().info(f'Spawning robot USD from {host_path}')
        return self.spawn_entity(f"file://{host_path}", entity_name, initial_pose)


def main():
    """Main entry point: load field USD and spawn robot from USD file."""
    rclpy.init()
    sim = SimulationInterface()

    try:
        # Load field USD
        sim.get_logger().info("Loading field1.usd...")
        if not sim.load_field_usd():
            sim.get_logger().error("Failed to load field USD")
            return 1
        time.sleep(1.0)

        # Spawn robot from USD
        robot_pose = PoseStamped()
        robot_pose.pose.position = Point(x=8.128949212924178,y=16.63174225312982,z=0.0820257550378943)
        robot_pose.pose.orientation = Quaternion(w=0.0,x=0.0,y=0.0,z=-1.0)
        sim.get_logger().info("Spawning robot from USD...")
        if not sim.spawn_robot_from_usd(initial_pose=robot_pose):
            sim.get_logger().error("Failed to spawn robot")
            return 1

        # Start simulation
        sim.get_logger().info("Starting simulation...")
        sim.set_simulation_state(SimulationState.STATE_PLAYING)

        sim.get_logger().info("Field and robot spawned successfully!")

        # Keep node alive
        rclpy.spin(sim)

    except KeyboardInterrupt:
        sim.get_logger().info("Shutting down...")
    finally:
        sim.destroy_node()
        rclpy.shutdown()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())