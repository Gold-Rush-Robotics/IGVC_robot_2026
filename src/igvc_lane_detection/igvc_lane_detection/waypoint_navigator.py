"""
igvc_waypoint_navigator.py  (revised)

Manages the GPS waypoint queue for IGVC AutoNav.  Rather than sending all
waypoints to Nav2 at once, it publishes the *active* waypoint to
/current_waypoint so LocalProgressNode can use it for bearing guidance.

The LocalProgressNode calls /waypoint_advance (std_srvs/Trigger) when the
robot is within xy_goal_tolerance of the current waypoint.

Waypoints file (waypoints.yaml):
    waypoints:
      - { lat: 42.6789, lon: -83.2134 }
      - { lat: 42.6791, lon: -83.2130 }

Usage:
    ros2 run your_pkg igvc_waypoint_navigator \
        --ros-args -p waypoints_file:=/path/to/waypoints.yaml
"""

import math
import yaml

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import NavSatFix
from std_srvs.srv import Trigger

_A  = 6378137.0
_E2 = 0.00669437999014


def _ecef(lat_deg, lon_deg, alt=0.0):
    lat = math.radians(lat_deg)
    lon = math.radians(lon_deg)
    N   = _A / math.sqrt(1 - _E2 * math.sin(lat) ** 2)
    return ((N + alt) * math.cos(lat) * math.cos(lon),
            (N + alt) * math.cos(lat) * math.sin(lon),
            (N * (1 - _E2) + alt) * math.sin(lat))


def gps_to_local(lat, lon, origin_lat, origin_lon):
    ox, oy, oz = _ecef(origin_lat, origin_lon)
    px, py, pz = _ecef(lat, lon)
    dx, dy, dz = px - ox, py - oy, pz - oz
    lat0 = math.radians(origin_lat)
    lon0 = math.radians(origin_lon)
    east  = -math.sin(lon0) * dx + math.cos(lon0) * dy
    north = (-math.sin(lat0) * math.cos(lon0) * dx
             - math.sin(lat0) * math.sin(lon0) * dy
             + math.cos(lat0) * dz)
    return east, north


class IGVCWaypointNavigator(Node):
    def __init__(self):
        super().__init__('igvc_waypoint_navigator')

        def p(name, default):
            self.declare_parameter(name, default)
            return self.get_parameter(name).value

        waypoints_file   = p('waypoints_file',  '')
        self.loop        = p('loop_waypoints',   False)
        self.map_frame   = p('map_frame',        'map')
        self.gps_topic   = p('gps_topic',        '/gps/fix')
        self.origin_lat  = p('origin_lat',        0.0)
        self.origin_lon  = p('origin_lon',        0.0)
        self._origin_set = (self.origin_lat != 0.0 or self.origin_lon != 0.0)

        self.waypoints_ll = self._load(waypoints_file)
        self.wp_index     = 0

        # Publishes the currently active waypoint in map frame
        self.wp_pub = self.create_publisher(PoseStamped, '/current_waypoint', 10)

        # LocalProgressNode calls this when the waypoint is reached
        self.create_service(Trigger, '/waypoint_advance', self._on_advance)

        # Republish current waypoint at 1 Hz so late-joining nodes get it
        self.create_timer(1.0, self._republish)

        self.create_subscription(NavSatFix, self.gps_topic, self._on_gps, 10)

        self.get_logger().info(
            f'Loaded {len(self.waypoints_ll)} waypoints from "{waypoints_file}"')

    # ── File loader ───────────────────────────────────────────────────────

    def _load(self, path):
        if not path:
            self.get_logger().warn('No waypoints_file set.')
            return []
        try:
            with open(path) as f:
                data = yaml.safe_load(f)
            return [(float(w['lat']), float(w['lon']))
                    for w in data.get('waypoints', [])]
        except Exception as e:
            self.get_logger().error(f'Failed to load waypoints: {e}')
            return []

    # ── GPS origin ────────────────────────────────────────────────────────

    def _on_gps(self, msg: NavSatFix):
        if not self._origin_set and msg.status.status >= 0:
            self.origin_lat  = msg.latitude
            self.origin_lon  = msg.longitude
            self._origin_set = True
            self.get_logger().info(
                f'Origin set: ({self.origin_lat:.6f}, {self.origin_lon:.6f})')
            self._publish_current()

    # ── Waypoint advance service ──────────────────────────────────────────

    def _on_advance(self, _request, response):
        if not self.waypoints_ll:
            response.success = False
            response.message = 'No waypoints loaded.'
            return response

        self.wp_index += 1
        if self.wp_index >= len(self.waypoints_ll):
            if self.loop:
                self.wp_index = 0
                self.get_logger().info('All waypoints done — looping.')
            else:
                self.wp_index = len(self.waypoints_ll) - 1
                self.get_logger().info('All waypoints complete.')
                response.success = True
                response.message = 'Course complete.'
                return response

        self.get_logger().info(
            f'Advanced to waypoint {self.wp_index + 1}/{len(self.waypoints_ll)}')
        self._publish_current()
        response.success = True
        response.message = f'Now targeting waypoint {self.wp_index}'
        return response

    # ── Publishers ────────────────────────────────────────────────────────

    def _publish_current(self):
        if not self._origin_set or not self.waypoints_ll:
            return
        lat, lon = self.waypoints_ll[self.wp_index]
        x, y = gps_to_local(lat, lon, self.origin_lat, self.origin_lon)

        ps = PoseStamped()
        ps.header.stamp    = self.get_clock().now().to_msg()
        ps.header.frame_id = self.map_frame
        ps.pose.position.x = x
        ps.pose.position.y = y
        ps.pose.orientation.w = 1.0
        self.wp_pub.publish(ps)

        self.get_logger().info(
            f'WP {self.wp_index + 1}/{len(self.waypoints_ll)}: '
            f'({lat:.6f}, {lon:.6f}) -> map ({x:.2f}, {y:.2f})',
            throttle_duration_sec=2.0)

    def _republish(self):
        self._publish_current()


def main(args=None):
    rclpy.init(args=args)
    rclpy.spin(IGVCWaypointNavigator())
    rclpy.shutdown()


if __name__ == '__main__':
    main()