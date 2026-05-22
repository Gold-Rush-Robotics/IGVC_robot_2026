#!/usr/bin/env python3
"""
Simple helper node to start simulation after robot is spawned.

This waits for the robot entity to exist, then calls SetSimulationState
to start the simulation playing. This ensures the field and robot are
loaded before the simulation starts.
"""

import rclpy
from rclpy.node import Node
import time
from simulation_interfaces.srv import GetEntityState, SetSimulationState
from simulation_interfaces.msg import Result, SimulationState


class SimStartupHelper(Node):
    def __init__(self):
        super().__init__('sim_startup_helper')
        
        self.declare_parameter('robot_entity_name', 'igvc_robot')
        self.declare_parameter('max_wait_seconds', 30.0)
        
        robot_entity_name = self.get_parameter('robot_entity_name').value
        max_wait = self.get_parameter('max_wait_seconds').value
        
        self.get_logger().info(f'Waiting for robot "{robot_entity_name}" to be spawned...')
        
        # Create service clients
        get_entity_client = self.create_client(GetEntityState, '/get_entity_state')
        set_sim_state_client = self.create_client(SetSimulationState, '/set_simulation_state')
        
        # Wait for services to be available
        if not get_entity_client.wait_for_service(timeout_sec=10.0):
            self.get_logger().error('GetEntityState service not available')
            return
        
        if not set_sim_state_client.wait_for_service(timeout_sec=10.0):
            self.get_logger().error('SetSimulationState service not available')
            return
        
        # Poll for robot entity
        start_time = time.time()
        while time.time() - start_time < max_wait:
            req = GetEntityState.Request()
            req.entity = robot_entity_name
            
            try:
                future = get_entity_client.call_async(req)
                rclpy.spin_until_future_complete(self, future, timeout_sec=1.0)
                
                if future.done() and future.result():
                    result = future.result()
                    if result.result.result == Result.RESULT_OK:
                        self.get_logger().info(f'Robot "{robot_entity_name}" found! Starting simulation...')
                        
                        # Start simulation
                        sim_req = SetSimulationState.Request()
                        sim_req.state = SimulationState.STATE_PLAYING
                        sim_future = set_sim_state_client.call_async(sim_req)
                        rclpy.spin_until_future_complete(self, sim_future, timeout_sec=5.0)
                        
                        if (sim_future.done() and sim_future.result() and
                                sim_future.result().result.result in (
                                    Result.RESULT_OK,
                                    SetSimulationState.Response.ALREADY_IN_TARGET_STATE,
                                )):
                            self.get_logger().info('Simulation started successfully')
                            return
                        else:
                            self.get_logger().error('Failed to start simulation')
                            return
            except Exception as e:
                self.get_logger().debug(f'Waiting for robot... ({e})')
            
            time.sleep(0.5)
        
        self.get_logger().error(f'Timeout waiting for robot "{robot_entity_name}"')


def main(args=None):
    rclpy.init(args=args)
    node = SimStartupHelper()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
