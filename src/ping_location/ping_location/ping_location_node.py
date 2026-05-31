"""ROS 2 node for fetching GPS data from remote phone endpoints and publishing NavSatFix messages.

This node connects to remote GPS endpoints via HTTP, retrieves GPS coordinates,
computes the midpoint between two phones, and publishes the data as NavSatFix messages.
"""
import requests
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import NavSatFix, NavSatStatus
from std_msgs.msg import Header


class PingLocationNode(Node):
    """Node for aggregating GPS data from multiple phone endpoints.
    
    Fetches GPS coordinates from two phone endpoints, computes their midpoint,
    and publishes all three as NavSatFix messages.
    """

    def __init__(self):
        super().__init__('ping_location_node')
        
        # Declare parameters with defaults
        self.declare_parameter('phone_1_ip', '192.168.0.162')
        self.declare_parameter('phone_2_ip', '192.168.0.164')
        self.declare_parameter('port', 8080)
        self.declare_parameter('timeout', 5)
        self.declare_parameter('loop_rate', 1)
        
        # Get parameters
        self.phone_1_ip = self.get_parameter('phone_1_ip').value
        self.phone_2_ip = self.get_parameter('phone_2_ip').value
        self.port = self.get_parameter('port').value
        self.timeout = self.get_parameter('timeout').value
        loop_rate = self.get_parameter('loop_rate').value
        
        # Create publishers
        self.phone1_pub = self.create_publisher(NavSatFix, 'ping_location/phone1', 10)
        self.phone2_pub = self.create_publisher(NavSatFix, 'ping_location/phone2', 10)
        self.midpoint_pub = self.create_publisher(NavSatFix, 'ping_location/midpoint', 10)
        
        # Create a timer to run the GPS fetch loop
        self.timer = self.create_timer(1.0 / loop_rate, self.gps_callback)
        
        self.get_logger().info(f'PingLocationNode started')
        self.get_logger().info(f'Phone 1 IP: {self.phone_1_ip}')
        self.get_logger().info(f'Phone 2 IP: {self.phone_2_ip}')

    def gps_url(self, ip: str) -> str:
        return f"http://{ip}:{self.port}/get?lat&lon&dir"

    def extract_value(self, buffer_data: dict, key: str) -> float | None:
        try:
            val = buffer_data[key]["buffer"][0]
            return float(val) if val is not None else None
        except (KeyError, IndexError, TypeError):
            return None

    def get_gps(self, ip: str) -> dict | None:
        """Fetch GPS data from a remote endpoint.
        
        Args:
            ip: IP address of the GPS endpoint
            
        Returns:
            Dictionary with 'lat', 'lon', and optionally 'dir' keys, or None on error
        """
        try:
            response = requests.get(self.gps_url(ip), timeout=self.timeout)
            response.raise_for_status()
            data = response.json()["buffer"]
            dir_val = self.extract_value(data, "dir")
            
            result = {
                "lat": self.extract_value(data, "lat"),
                "lon": self.extract_value(data, "lon"),
            }
            
            # Only include dir if it's a valid number
            if dir_val is not None:
                result["dir"] = dir_val
                
            return result
        except requests.exceptions.Timeout:
            self.get_logger().debug(f"[{ip}] GPS request timeout")
            return None
        except requests.exceptions.ConnectionError:
            self.get_logger().debug(f"[{ip}] GPS connection error")
            return None
        except (requests.exceptions.RequestException, ValueError, KeyError) as e:
            self.get_logger().debug(f"[{ip}] GPS error: {e}")
            return None

    def midpoint(self, data1: dict, data2: dict) -> dict | None:
        if data1["lat"] is None or data2["lat"] is None:
            self.get_logger().debug("Cannot compute midpoint — missing lat/lon from one or both phones")
            return None
        
        result = {
            "lat": (data1["lat"] + data2["lat"]) / 2,
            "lon": (data1["lon"] + data2["lon"]) / 2,
        }
        
        # Only include dir if both phones have valid direction values
        if "dir" in data1 and "dir" in data2:
            result["dir"] = (data1["dir"] + data2["dir"]) / 2
        
        return result

    def create_navsatfix_msg(self, gps_data: dict, frame_id: str = "gps") -> NavSatFix:
        """Create a NavSatFix message from GPS data dictionary."""
        msg = NavSatFix()
        
        # Set header
        msg.header = Header()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = frame_id
        
        # Set status
        msg.status = NavSatStatus()
        msg.status.status = NavSatStatus.STATUS_FIX
        msg.status.service = NavSatStatus.SERVICE_GPS
        
        # Set position
        msg.latitude = gps_data["lat"]
        msg.longitude = gps_data["lon"]
        msg.altitude = float('nan')  # Not provided by GPS endpoint
        
        # Set position covariance (9 elements for 3x3 matrix)
        msg.position_covariance = [0.0] * 9
        msg.position_covariance_type = NavSatFix.COVARIANCE_TYPE_UNKNOWN
        
        return msg

    def gps_callback(self):
        phone1 = self.get_gps(self.phone_1_ip)
        phone2 = self.get_gps(self.phone_2_ip)

        # Publish phone 1 data
        if phone1 and phone1["lat"] is not None and phone1["lon"] is not None:
            msg = self.create_navsatfix_msg(phone1, frame_id="phone1_gps")
            self.phone1_pub.publish(msg)

        # Publish phone 2 data
        if phone2 and phone2["lat"] is not None and phone2["lon"] is not None:
            msg = self.create_navsatfix_msg(phone2, frame_id="phone2_gps")
            self.phone2_pub.publish(msg)

        # Compute and publish midpoint
        if phone1 and phone2:
            mid = self.midpoint(phone1, phone2)
            if mid and mid["lat"] is not None and mid["lon"] is not None:
                msg = self.create_navsatfix_msg(mid, frame_id="midpoint_gps")
                self.midpoint_pub.publish(msg)
                
                dir_str = f" Dir: {mid['dir']}°" if "dir" in mid else ""
                self.get_logger().info(f"Midpoint | Lat: {mid['lat']:.7f}  Lon: {mid['lon']:.7f}{dir_str}")


def main(args=None):
    rclpy.init(args=args)
    
    node = PingLocationNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
