"""
igvc_local_progress.py

Bridges the gap between sparse GPS waypoints and Nav2's local controller.
Instead of asking the global planner to route across the entire course,
this node publishes a short-horizon local goal that keeps the robot moving
toward the current GPS waypoint while respecting the lane.

Two complementary outputs:
  1. /local_goal  (PoseStamped) — a "carrot" 2–3 m ahead along the bearing
     to the current GPS waypoint.  Nav2's NavigateToPose action tracks this.
     Updated every frame so the carrot slides forward as the robot moves.

  2. /lane_path   (Path) — a dense centerline path built from the lane
     detection costmap.  Published for RegulatedPurePursuit to follow
     reactively.  Waypoint bearing is blended in when the lane curves away
     from the goal.

The waypoint follower (igvc_waypoint_navigator.py) feeds GPS waypoints via
the /current_waypoint topic (PoseStamped in map frame).  When a waypoint is
reached (within xy_goal_tolerance), it requests the next one via the
/waypoint_advance service.

Subscriptions
  /current_waypoint  (PoseStamped)   — active GPS waypoint in map frame
  /lane_costmap      (OccupancyGrid) — lane boundaries from lane_detection_node
  /odom              (Odometry)      — robot pose for carrot projection

Publications
  /local_goal        (PoseStamped)   — short-horizon carrot goal
  /lane_path         (Path)          — centerline path for RPP
  /progress_debug    (Image)         — optional top-down visualisation
"""

import math
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.time import Time
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy

from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import OccupancyGrid, Odometry, Path
from sensor_msgs.msg import Image
from std_srvs.srv import Trigger

from tf2_ros import Buffer, TransformListener
import tf2_geometry_msgs  # noqa: F401 — registers PoseStamped transform support

from cv_bridge import CvBridge
import cv2


class LocalProgressNode(Node):

    def __init__(self):
        super().__init__('igvc_local_progress')
        self.bridge = CvBridge()

        def p(name, default):
            self.declare_parameter(name, default)
            return self.get_parameter(name).value

        # ── Parameters ────────────────────────────────────────────────────
        self.map_frame          = p('map_frame',              'map')
        self.odom_frame         = p('odom_frame',             'odom')
        self.base_frame         = p('base_frame',             'base_link')
        self.carrot_dist        = p('carrot_distance_m',       2.5)   # how far ahead to place goal
        self.goal_tolerance     = p('xy_goal_tolerance_m',     0.8)   # when to call WP reached
        self.lane_blend_weight  = p('lane_blend_weight',       0.6)   # 0=pure GPS, 1=pure lane
        self.grid_res           = p('grid_resolution',         0.05)
        self.grid_width_m       = p('grid_width_m',           10.0)
        self.grid_height_m      = p('grid_height_m',          10.0)
        self.publish_debug      = p('publish_debug_image',     True)
        self.path_lookahead_m   = p('path_lookahead_m',        4.0)   # how far ahead to build path

        # ── State ─────────────────────────────────────────────────────────
        self.current_waypoint: PoseStamped | None = None
        self.latest_grid: OccupancyGrid | None    = None
        self.robot_pose: PoseStamped | None       = None

        # ── TF ────────────────────────────────────────────────────────────
        self.tf_buffer   = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # ── Subscribers ───────────────────────────────────────────────────
        self.create_subscription(PoseStamped,   '/current_waypoint', self._on_waypoint,  10)
        self.create_subscription(OccupancyGrid, '/lane_costmap',     self._on_grid,      10)
        self.create_subscription(Odometry,      '/odom',             self._on_odom,      10)

        # ── Publishers ────────────────────────────────────────────────────
        self.goal_pub  = self.create_publisher(PoseStamped, '/local_goal',  10)
        self.path_pub  = self.create_publisher(Path,        '/lane_path',   10)
        if self.publish_debug:
            self.debug_pub = self.create_publisher(Image, '/progress_debug', 10)

        # ── Waypoint advance service (called when WP reached) ─────────────
        self.advance_client = self.create_client(Trigger, '/waypoint_advance')

        # ── Main loop ─────────────────────────────────────────────────────
        self.create_timer(0.1, self._update)   # 10 Hz

        self.get_logger().info('LocalProgressNode ready.')

    # ── Callbacks ─────────────────────────────────────────────────────────

    def _on_waypoint(self, msg: PoseStamped):
        self.current_waypoint = msg
        self.get_logger().info(
            f'New waypoint: ({msg.pose.position.x:.2f}, {msg.pose.position.y:.2f})',
            throttle_duration_sec=2.0)

    def _on_grid(self, msg: OccupancyGrid):
        self.latest_grid = msg

    def _on_odom(self, msg: Odometry):
        ps = PoseStamped()
        ps.header = msg.header
        ps.pose   = msg.pose.pose
        self.robot_pose = ps

    # ── Main update ───────────────────────────────────────────────────────

    def _update(self):
        if self.robot_pose is None:
            self.get_logger().warn('Waiting for odometry...', throttle_duration_sec=3.0)
            return

        # Transform robot pose to map frame for waypoint comparison
        robot_in_map = self._transform_pose(self.robot_pose, self.map_frame)
        if robot_in_map is None:
            return

        # Check if current waypoint is reached
        if self.current_waypoint is not None:
            dist = self._dist(robot_in_map, self.current_waypoint)
            if dist < self.goal_tolerance:
                self.get_logger().info(f'Waypoint reached (dist={dist:.2f}m), advancing.')
                self._advance_waypoint()

        # Build lane centerline in base_link frame
        lane_center_pts = self._extract_lane_centerline()

        # Compute GPS bearing in base_link frame
        gps_bearing_local = self._gps_bearing_in_base_link(robot_in_map)

        # Build and publish the path
        path = self._build_path(lane_center_pts, gps_bearing_local)
        self.path_pub.publish(path)

        # Publish carrot goal
        carrot = self._make_carrot(lane_center_pts, gps_bearing_local)
        if carrot is not None:
            self.goal_pub.publish(carrot)

        if self.publish_debug:
            self._publish_debug(lane_center_pts, gps_bearing_local, carrot)

    # ── Lane centerline extraction ─────────────────────────────────────────

    def _extract_lane_centerline(self):
        """
        Read the /lane_costmap occupancy grid and extract the centreline of
        the free (cost=0) corridor for each row ahead of the robot.

        Returns a list of (forward_m, lateral_m) tuples in base_link frame,
        sorted by forward distance.  Returns [] if no grid or no free cells.
        """
        g = self.latest_grid
        if g is None:
            return []

        W, H   = g.info.width, g.info.height
        res    = g.info.resolution
        orig_y = g.info.origin.position.y   # = -grid_width/2

        data = np.array(g.data, dtype=np.int8).reshape(H, W)
        pts  = []

        for row in range(H):
            fwd = row * res
            if fwd > self.path_lookahead_m:
                break

            # Find columns marked free (0) in this row
            free_cols = np.where(data[row] == 0)[0]
            if len(free_cols) == 0:
                continue

            # Centroid of free cells = lane centre at this distance
            centre_col = float(np.mean(free_cols))
            lateral    = orig_y + centre_col * res   # metres, left positive
            pts.append((fwd, lateral))

        return pts

    # ── GPS bearing in base_link ───────────────────────────────────────────

    def _gps_bearing_in_base_link(self, robot_in_map: PoseStamped):
        """
        Returns the unit vector (dx, dy) pointing from the robot toward the
        current GPS waypoint, expressed in base_link frame.
        Returns (1.0, 0.0) (straight ahead) if no waypoint available.
        """
        if self.current_waypoint is None:
            return (1.0, 0.0)

        wp = self.current_waypoint
        rx = robot_in_map.pose.position.x
        ry = robot_in_map.pose.position.y

        # Vector in map frame
        dx_map = wp.pose.position.x - rx
        dy_map = wp.pose.position.y - ry
        dist   = math.hypot(dx_map, dy_map)
        if dist < 1e-3:
            return (1.0, 0.0)

        dx_map /= dist
        dy_map /= dist

        # Rotate into base_link frame using robot yaw
        yaw = self._yaw_from_pose(robot_in_map)
        cos_y, sin_y = math.cos(yaw), math.sin(yaw)
        # Map->base_link rotation is transpose of base_link->map
        dx_base =  cos_y * dx_map + sin_y * dy_map
        dy_base = -sin_y * dx_map + cos_y * dy_map

        return (dx_base, dy_base)

    # ── Path builder ──────────────────────────────────────────────────────

    def _build_path(self, lane_pts, gps_bearing):
        """
        Build a Path message by blending lane centerline points with the GPS
        bearing direction.  When lane_pts is sparse or absent, falls back to
        a straight line toward the GPS waypoint.
        """
        path = Path()
        path.header.stamp    = self.get_clock().now().to_msg()
        path.header.frame_id = self.base_frame

        if not lane_pts:
            # Pure GPS bearing fallback: project a few waypoints straight ahead
            bearing_angle = math.atan2(gps_bearing[1], gps_bearing[0])
            for d in np.linspace(0.5, self.path_lookahead_m, 10):
                ps = PoseStamped()
                ps.header = path.header
                ps.pose.position.x = d * math.cos(bearing_angle)
                ps.pose.position.y = d * math.sin(bearing_angle)
                ps.pose.orientation.z = math.sin(bearing_angle / 2)
                ps.pose.orientation.w = math.cos(bearing_angle / 2)
                path.poses.append(ps)
            return path

        bearing_angle = math.atan2(gps_bearing[1], gps_bearing[0])

        for fwd, lat in lane_pts:
            # Lane-only lateral position
            lane_lateral = lat

            # GPS-bearing lateral position at this forward distance
            gps_lateral = fwd * math.tan(bearing_angle)

            # Blend: use lane heavily when we have good coverage, GPS when not
            blended_lat = (self.lane_blend_weight * lane_lateral
                           + (1.0 - self.lane_blend_weight) * gps_lateral)

            ps = PoseStamped()
            ps.header = path.header
            ps.pose.position.x = fwd
            ps.pose.position.y = blended_lat
            ps.pose.orientation.w = 1.0
            path.poses.append(ps)

        # Fill in yaw from consecutive points
        for i, ps in enumerate(path.poses[:-1]):
            dx = path.poses[i + 1].pose.position.x - ps.pose.position.x
            dy = path.poses[i + 1].pose.position.y - ps.pose.position.y
            yaw = math.atan2(dy, dx)
            ps.pose.orientation.z = math.sin(yaw / 2)
            ps.pose.orientation.w = math.cos(yaw / 2)

        return path

    # ── Carrot goal ───────────────────────────────────────────────────────

    def _make_carrot(self, lane_pts, gps_bearing):
        """
        Place a carrot goal `carrot_dist` metres ahead along the blended path.
        Published in base_link frame — Nav2 will transform it to map internally.
        """
        path = self._build_path(lane_pts, gps_bearing)
        if not path.poses:
            return None

        # Find the pose closest to carrot_dist forward
        best = None
        for ps in path.poses:
            d = math.hypot(ps.pose.position.x, ps.pose.position.y)
            if d >= self.carrot_dist:
                best = ps
                break
        if best is None:
            best = path.poses[-1]

        carrot = PoseStamped()
        carrot.header.stamp    = self.get_clock().now().to_msg()
        carrot.header.frame_id = self.base_frame
        carrot.pose            = best.pose
        return carrot

    # ── Waypoint advance ──────────────────────────────────────────────────

    def _advance_waypoint(self):
        if not self.advance_client.service_is_ready():
            self.get_logger().warn('/waypoint_advance service not ready.')
            return
        self.advance_client.call_async(Trigger.Request())

    # ── Debug image ───────────────────────────────────────────────────────

    def _publish_debug(self, lane_pts, gps_bearing, carrot):
        """Top-down 200x200 occupancy grid view with overlaid path and carrot."""
        scale = 10   # pixels per cell at grid_res
        W = int(self.grid_width_m  / self.grid_res)
        H = int(self.grid_height_m / self.grid_res)
        img = np.full((H * scale // 10, W * scale // 10, 3), 40, dtype=np.uint8)

        if self.latest_grid is not None:
            data = np.array(self.latest_grid.data, dtype=np.int8).reshape(H, W)
            vis  = cv2.resize(
                data.astype(np.float32), (W * scale // 10, H * scale // 10),
                interpolation=cv2.INTER_NEAREST)
            img[vis == 0]   = [60, 100, 60]    # free = dark green
            img[vis == 100] = [200, 80,  80]   # lane = red

        ch, cw = img.shape[:2]
        orig_y = -self.grid_width_m / 2.0

        def to_px(fwd, lat):
            px = int(fwd  / self.grid_height_m * ch)
            py = int((lat - orig_y) / self.grid_width_m * cw)
            return (py, ch - 1 - px)

        # Draw lane centerline
        for i in range(len(lane_pts) - 1):
            cv2.line(img, to_px(*lane_pts[i]), to_px(*lane_pts[i + 1]),
                     (100, 200, 100), 1)

        # Draw GPS bearing line
        bearing_angle = math.atan2(gps_bearing[1], gps_bearing[0])
        gx = to_px(0, 0)
        gx2 = to_px(self.path_lookahead_m,
                    self.path_lookahead_m * math.tan(bearing_angle))
        cv2.line(img, gx, gx2, (80, 80, 200), 1)

        # Draw carrot
        if carrot is not None:
            cp = to_px(carrot.pose.position.x, carrot.pose.position.y)
            cv2.circle(img, cp, 4, (255, 200, 0), -1)

        # Robot position
        cv2.circle(img, to_px(0, 0), 5, (255, 255, 255), -1)

        try:
            msg = self.bridge.cv2_to_imgmsg(
                cv2.flip(img, 0), 'bgr8')
            msg.header.stamp    = self.get_clock().now().to_msg()
            msg.header.frame_id = self.base_frame
            self.debug_pub.publish(msg)
        except Exception as e:
            self.get_logger().warn(f'Debug image error: {e}', throttle_duration_sec=2.0)

    # ── Helpers ───────────────────────────────────────────────────────────

    def _transform_pose(self, pose: PoseStamped, target_frame: str):
        try:
            return self.tf_buffer.transform(pose, target_frame,
                                            timeout=Duration(seconds=0.05))
        except Exception:
            try:
                return self.tf_buffer.transform(pose, target_frame,
                                                timeout=Duration(seconds=0.1))
            except Exception as e:
                self.get_logger().warn(
                    f'TF {pose.header.frame_id}->{target_frame}: {e}',
                    throttle_duration_sec=2.0)
                return None

    @staticmethod
    def _dist(a: PoseStamped, b: PoseStamped):
        return math.hypot(a.pose.position.x - b.pose.position.x,
                          a.pose.position.y - b.pose.position.y)

    @staticmethod
    def _yaw_from_pose(pose: PoseStamped):
        q = pose.pose.orientation
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny_cosp, cosy_cosp)


def main(args=None):
    rclpy.init(args=args)
    rclpy.spin(LocalProgressNode())
    rclpy.shutdown()


if __name__ == '__main__':
    main()