import rclpy
from rclpy.action import ActionClient
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from rclpy.time import Time

from sensor_msgs.msg import Image, CameraInfo
from message_filters import Subscriber, ApproximateTimeSynchronizer
from nav_msgs.msg import Path, OccupancyGrid
from geometry_msgs.msg import PoseStamped
from nav2_msgs.action import FollowPath
from lifecycle_msgs.srv import GetState
from tf2_ros import Buffer, TransformListener

import cv2
import numpy as np
from cv_bridge import CvBridge

MIN_ABS_SLOPE = 1e-3


class LaneDetectionNode(Node):
    def __init__(self):
        super().__init__('lane_detection_node')
        self.bridge   = CvBridge()
        self.cam_mats = {}          # camera_index -> 3x3 K matrix
        self.observations = {}      # camera_index -> {stamp, left, right}

        # ---------------------------------------------------------------- #
        # Parameters                                                         #
        # ---------------------------------------------------------------- #
        def p(name, default):
            self.declare_parameter(name, default)
            return self.get_parameter(name).value

        self.lane_fusion_frame    = p('lane_fusion_frame',          'base_link')
        self.occ_grid_frame       = p('occupancy_grid_frame',       '') or self.lane_fusion_frame
        self.grid_res             = p('grid_resolution',             0.05)   # m/cell
        self.grid_width_m         = p('grid_width',                 10.0)   # m
        self.grid_height_m        = p('grid_height',                10.0)   # m
        self.fusion_timeout_sec   = p('fusion_timeout_sec',          0.5)
        self.depth_scale          = p('depth_scale',                 1.0)   # set 0.001 for mm floats
        self.assumed_lane_width_m = p('assumed_lane_width_m',        3.5)   # fallback when one boundary missing
        self.publish_overlay      = p('publish_overlay_debug',       True)
        self.overlay_prefix       = p('overlay_topic_prefix',       '/lane_debug')
        self.enable_follow_path   = p('enable_follow_path_action',   True)
        self.follow_path_name     = p('follow_path_action_name',    'follow_path')
        self.controller_id        = p('follow_path_controller_id',  'FollowPath')
        self.goal_checker_id      = p('follow_path_goal_checker_id','goal_checker')
        self.min_goal_period_sec  = p('min_action_send_period_sec',  0.25)
        self.required_tf_frame    = p('required_tf_target_frame',   'odom')
        self.require_ctrl_active  = p('require_controller_active',   False)  # set True only if you need lifecycle guard
        num_cameras   = p('num_cameras',         1)
        cam_topics    = p('camera_topics',       ['/camera/image_raw'])
        depth_topics  = p('depth_topics',        ['/camera/depth'])
        info_topics   = p('camera_info_topics',  ['/camera/camera_info'])
        path_topic    = p('lane_center_debug_topic', '/lane_center_debug')
        grid_topic    = p('occupancy_grid_topic',    '/lane_costmap')

        num_cameras = min(num_cameras, len(cam_topics), len(depth_topics), len(info_topics))
        self.get_logger().info(f'LaneDetectionNode: {num_cameras} camera(s), '
                               f'fusion_frame={self.lane_fusion_frame}')

        # ---------------------------------------------------------------- #
        # TF                                                                 #
        # ---------------------------------------------------------------- #
        self.tf_buffer   = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # ---------------------------------------------------------------- #
        # Subscriptions                                                      #
        # ---------------------------------------------------------------- #
        self.overlay_pubs  = {}
        self._sync_handles = []  # keep alive

        for i in range(num_cameras):
            self.create_subscription(
                CameraInfo, info_topics[i],
                lambda msg, idx=i: self._on_camera_info(msg, idx), 10)

            rgb_sub   = Subscriber(self, Image, cam_topics[i])
            depth_sub = Subscriber(self, Image, depth_topics[i])
            sync = ApproximateTimeSynchronizer([rgb_sub, depth_sub], queue_size=10, slop=0.1)
            sync.registerCallback(lambda r, d, idx=i: self._on_synced_images(r, d, idx))
            self._sync_handles.append((rgb_sub, depth_sub, sync))

            if self.publish_overlay:
                name  = cam_topics[i].split('/')[1] if cam_topics[i].count('/') >= 2 else f'cam{i}'
                topic = f'{self.overlay_prefix}/{name}/overlay'
                self.overlay_pubs[i] = self.create_publisher(Image, topic, 10)
                self.get_logger().info(f'  camera[{i}]: rgb={cam_topics[i]}  overlay={topic}')

        # ---------------------------------------------------------------- #
        # Publishers                                                         #
        # ---------------------------------------------------------------- #
        map_qos = QoSProfile(reliability=ReliabilityPolicy.RELIABLE,
                             durability=DurabilityPolicy.TRANSIENT_LOCAL,
                             history=HistoryPolicy.KEEP_LAST, depth=1)

        self.path_pub    = self.create_publisher(Path,          path_topic, 10)
        self.grid_pub    = self.create_publisher(OccupancyGrid, grid_topic, map_qos)
        self.latest_grid = self._empty_grid()
        # Periodically republish last known grid (keeps costmap alive)
        self.create_timer(1.0, lambda: (
            setattr(self.latest_grid.header, 'stamp', self.get_clock().now().to_msg()),
            self.grid_pub.publish(self.latest_grid)))

        # ---------------------------------------------------------------- #
        # FollowPath action state                                            #
        # ---------------------------------------------------------------- #
        self.fp_client            = ActionClient(self, FollowPath, self.follow_path_name)
        self.goal_pending         = False
        self.goal_active          = False
        self.current_goal_handle  = None
        self.last_goal_time       = None

        # ---------------------------------------------------------------- #
        # Controller lifecycle state (optional guard)                        #
        # ---------------------------------------------------------------- #
        self.ctrl_active          = False
        self.ctrl_state_label     = 'unknown'
        self._ctrl_req_pending    = False
        self.ctrl_client          = self.create_client(GetState, 'controller_server/get_state')
        self.create_timer(1.0, self._poll_controller_state)

        # ---------------------------------------------------------------- #
        # Watchdog                                                           #
        # ---------------------------------------------------------------- #
        self._got_frame = False
        self.create_timer(2.0, self._watchdog)

    # ==================================================================== #
    #  Camera info                                                           #
    # ==================================================================== #

    def _on_camera_info(self, msg, idx):
        self.cam_mats[idx] = np.array(msg.k).reshape(3, 3)
        self.get_logger().info(f'Intrinsics received for camera[{idx}].', once=True)

    # ==================================================================== #
    #  Main image callback                                                   #
    # ==================================================================== #

    def _on_synced_images(self, rgb_msg, depth_msg, cam_idx):
        self._got_frame = True

        # --- Decode ---
        try:
            bgr = self.bridge.imgmsg_to_cv2(rgb_msg, 'bgr8')
            if depth_msg.encoding == '16UC1':
                depth = self.bridge.imgmsg_to_cv2(depth_msg, '16UC1').astype(np.float32) / 1000.0
            else:
                depth = self.bridge.imgmsg_to_cv2(depth_msg, '32FC1') * self.depth_scale
        except Exception as e:
            self.get_logger().error(f'Image decode error: {e}')
            return

        # --- Detect 2-D lane lines ---
        left_px, right_px, hough = self._detect_lanes(bgr)

        if self.publish_overlay and cam_idx in self.overlay_pubs:
            self._publish_overlay(cam_idx, bgr, hough, left_px, right_px, rgb_msg)

        # --- Project to 3-D (camera frame) ---
        left_3d  = self._line_to_3d(left_px,  depth, cam_idx)
        right_3d = self._line_to_3d(right_px, depth, cam_idx)

        self.get_logger().info(
            f'camera[{cam_idx}] hough={0 if hough is None else len(hough)} '
            f'left_px={left_px is not None} right_px={right_px is not None} '
            f'left_3d={0 if left_3d is None else len(left_3d)} '
            f'right_3d={0 if right_3d is None else len(right_3d)}',
            throttle_duration_sec=1.0)

        # --- Fill missing boundary using assumed lane width ---
        #     This is the key fallback: if the camera sees only one lane
        #     marking, we estimate the other at a fixed lateral offset.
        left_3d, right_3d = self._fill_missing_boundary(left_3d, right_3d)

        # --- Transform to fusion frame ---
        src   = rgb_msg.header.frame_id
        stamp = Time.from_msg(rgb_msg.header.stamp)
        self.observations[cam_idx] = {
            'stamp': self.get_clock().now(),
            'left':  self._transform_pts(left_3d,  src, self.lane_fusion_frame, stamp),
            'right': self._transform_pts(right_3d, src, self.lane_fusion_frame, stamp),
        }

        # --- Fuse observations from all cameras ---
        fused_left, fused_right = self._fuse()

        self.get_logger().info(
            f'fused left={0 if fused_left  is None else len(fused_left)} '
            f'right={0 if fused_right is None else len(fused_right)}',
            throttle_duration_sec=1.0)

        # --- Build and publish centerline path ---
        path = self._build_path(fused_left, fused_right, rgb_msg.header.stamp)
        self.path_pub.publish(path)
        self.get_logger().info(
            f'Path published: {len(path.poses)} poses, frame={path.header.frame_id}',
            throttle_duration_sec=1.0)

        # --- Build and publish occupancy grid ---
        self.latest_grid = self._build_grid(fused_left, fused_right, rgb_msg.header.stamp)
        self.grid_pub.publish(self.latest_grid)

        # --- Send nav2 goal ---
        self._send_goal(path)

    # ==================================================================== #
    #  Lane detection (image space)                                          #
    # ==================================================================== #

    def _detect_lanes(self, bgr):
        gray  = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        blur  = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 50, 150)
        roi   = self._apply_roi(edges)
        hough = cv2.HoughLinesP(roi, 1, np.pi/180,
                                threshold=20, minLineLength=20, maxLineGap=500)
        if hough is None:
            return None, None, None

        left_px, right_px = self._fit_lines(bgr, hough)
        return left_px, right_px, hough

    def _apply_roi(self, img):
        """Trapezoidal region of interest focused on the road ahead."""
        mask = np.zeros_like(img)
        r, c = img.shape[:2]
        pts  = np.array([[
            [c * 0.1, r * 0.95], [c * 0.4, r * 0.6],
            [c * 0.6, r * 0.6],  [c * 0.9, r * 0.95],
        ]], dtype=np.int32)
        cv2.fillPoly(mask, pts, 255)
        return cv2.bitwise_and(img, mask)

    def _fit_lines(self, img, hough):
        """
        Classify Hough segments as left/right by where they cross the image
        bottom, then compute a single length-weighted average line for each side.
        Returns pixel endpoint pairs: ((x1,y1),(x2,y2)) or None.
        """
        h, w = img.shape[:2]
        left_segs,  left_wts  = [], []
        right_segs, right_wts = [], []

        for seg in hough:
            x1, y1, x2, y2 = seg[0]
            if x1 == x2:
                continue
            slope = (y2 - y1) / (x2 - x1)
            if not np.isfinite(slope) or abs(slope) < MIN_ABS_SLOPE:
                continue
            intercept = y1 - slope * x1
            x_bottom  = (h - intercept) / slope
            length    = np.hypot(x2 - x1, y2 - y1)
            if x_bottom < w * 0.5:
                left_segs.append((slope, intercept));  left_wts.append(length)
            else:
                right_segs.append((slope, intercept)); right_wts.append(length)

        def avg_to_pixels(segs, wts):
            if not segs:
                return None
            s, b = np.average(segs, axis=0, weights=wts)
            if abs(s) < MIN_ABS_SLOPE:
                return None
            y1_ = int(h)
            y2_ = int(h * 0.6)
            return ((int((y1_ - b) / s), y1_), (int((y2_ - b) / s), y2_))

        return avg_to_pixels(left_segs, left_wts), avg_to_pixels(right_segs, right_wts)

    # ==================================================================== #
    #  3-D projection                                                        #
    # ==================================================================== #

    def _line_to_3d(self, line, depth, cam_idx, n_samples=20):
        """
        Sample points along a pixel line, look up depth, and back-project
        to 3-D in the camera optical frame (x right, y down, z forward).
        """
        if line is None:
            return None
        (x1, y1), (x2, y2) = line
        h, w = depth.shape[:2]
        K = self.cam_mats.get(cam_idx)
        fx, fy = (K[0, 0], K[1, 1]) if K is not None else (500., 500.)
        cx, cy = (K[0, 2], K[1, 2]) if K is not None else (320., 240.)

        pts = []
        for t in np.linspace(0.0, 1.0, n_samples):
            u = int(x1 + t * (x2 - x1))
            v = int(y1 + t * (y2 - y1))
            if not (0 <= u < w and 0 <= v < h):
                continue
            d = float(depth[v, u])
            if not (0.1 < d < 20.0):
                continue
            pts.append(((u - cx) * d / fx,
                         (v - cy) * d / fy,
                         d))
        return pts if pts else None

    def _fill_missing_boundary(self, left_3d, right_3d):
        """
        If only one lane boundary was detected, estimate the other by
        shifting laterally by `assumed_lane_width_m` in the camera x-axis
        (positive x = right in camera optical frame).
        """
        W = self.assumed_lane_width_m
        if left_3d is not None and right_3d is None:
            right_3d = [(x - W, y, z) for x, y, z in left_3d]
            self.get_logger().info('Right boundary estimated from left + lane width.',
                                   throttle_duration_sec=2.0)
        elif right_3d is not None and left_3d is None:
            left_3d  = [(x + W, y, z) for x, y, z in right_3d]
            self.get_logger().info('Left boundary estimated from right + lane width.',
                                   throttle_duration_sec=2.0)
        return left_3d, right_3d

    # ==================================================================== #
    #  TF transform                                                          #
    # ==================================================================== #

    def _transform_pts(self, points, src, tgt, stamp):
        if points is None or src == tgt:
            return points
        try:
            tf = self.tf_buffer.lookup_transform(tgt, src, stamp,
                                                 timeout=Duration(seconds=0.05))
        except Exception:
            try:
                tf = self.tf_buffer.lookup_transform(tgt, src, Time(),
                                                     timeout=Duration(seconds=0.1))
                self.get_logger().warn(f'TF fallback to latest: {tgt}<-{src}',
                                       throttle_duration_sec=2.0)
            except Exception as e:
                self.get_logger().warn(f'TF lookup failed {tgt}<-{src}: {e}',
                                       throttle_duration_sec=1.0)
                return None

        t = tf.transform.translation
        q = tf.transform.rotation
        R = self._quat_to_rot(q.x, q.y, q.z, q.w)
        tv = np.array([t.x, t.y, t.z])
        return [tuple(R @ np.array(p) + tv) for p in points]

    @staticmethod
    def _quat_to_rot(qx, qy, qz, qw):
        return np.array([
            [1 - 2*(qy*qy + qz*qz),  2*(qx*qy - qz*qw),  2*(qx*qz + qy*qw)],
            [    2*(qx*qy + qz*qw),  1 - 2*(qx*qx + qz*qz),  2*(qy*qz - qx*qw)],
            [    2*(qx*qz - qy*qw),  2*(qy*qz + qx*qw),  1 - 2*(qx*qx + qy*qy)],
        ])

    # ==================================================================== #
    #  Multi-camera fusion                                                   #
    # ==================================================================== #

    def _fuse(self):
        """Concatenate lane points from all cameras, dropping stale ones."""
        now = self.get_clock().now()
        stale = [k for k, v in self.observations.items()
                 if (now - v['stamp']).nanoseconds / 1e9 > self.fusion_timeout_sec]
        for k in stale:
            del self.observations[k]

        left, right = [], []
        for obs in self.observations.values():
            if obs['left']:  left.extend(obs['left'])
            if obs['right']: right.extend(obs['right'])

        return (left if left else None), (right if right else None)

    # ==================================================================== #
    #  Centerline path                                                       #
    # ==================================================================== #

    def _build_path(self, left_pts, right_pts, stamp):
        path = Path()
        path.header.stamp    = stamp
        path.header.frame_id = self.lane_fusion_frame

        if left_pts is None or right_pts is None:
            return path

        # Project onto (x, y) in fusion frame; sort by forward distance x
        lxy = sorted([(p[0], p[1]) for p in left_pts],  key=lambda p: p[0])
        rxy = sorted([(p[0], p[1]) for p in right_pts], key=lambda p: p[0])
        x_max = min(lxy[-1][0], rxy[-1][0], self.grid_height_m)

        if x_max <= 0.0:
            return path

        for x in np.linspace(0.0, x_max, 20):
            ly = self._interp_y(lxy, x)
            ry = self._interp_y(rxy, x)
            if ly is None or ry is None:
                continue
            ps = PoseStamped()
            ps.header.stamp    = stamp
            ps.header.frame_id = self.lane_fusion_frame
            ps.pose.position.x = float(x)
            ps.pose.position.y = 0.5 * (ly + ry)
            ps.pose.orientation.w = 1.0
            path.poses.append(ps)

        return path

    # ==================================================================== #
    #  Occupancy grid                                                        #
    # ==================================================================== #

    def _build_grid(self, left_pts, right_pts, stamp):
        grid = self._empty_grid(stamp)
        if left_pts is None or right_pts is None:
            return grid

        lxy = sorted([(p[0], p[1]) for p in left_pts],  key=lambda p: p[0])
        rxy = sorted([(p[0], p[1]) for p in right_pts], key=lambda p: p[0])
        if len(lxy) < 2 or len(rxy) < 2:
            return grid

        W    = grid.info.width
        H    = grid.info.height
        res  = self.grid_res
        data = [-1] * (W * H)

        for row in range(H):
            x  = row * res
            ly = self._interp_y(lxy, x)
            ry = self._interp_y(rxy, x)
            if ly is None or ry is None:
                continue
            if ly < ry:
                ly, ry = ry, ly  # ensure left > right in y

            lc = max(0, min(W - 1, int((ly + self.grid_width_m / 2) / res)))
            rc = max(0, min(W - 1, int((ry + self.grid_width_m / 2) / res)))

            for col in range(W):
                # 0 = free (inside lane), 100 = occupied (boundary or outside)
                data[row * W + col] = 0 if rc < col < lc else 100

        grid.data = data
        return grid

    def _empty_grid(self, stamp=None):
        g = OccupancyGrid()
        g.header.stamp    = self.get_clock().now().to_msg() if stamp is None else stamp
        g.header.frame_id = self.occ_grid_frame
        W = int(self.grid_width_m  / self.grid_res)
        H = int(self.grid_height_m / self.grid_res)
        g.info.resolution = self.grid_res
        g.info.width  = W
        g.info.height = H
        g.info.origin.position.x  =  0.0
        g.info.origin.position.y  = -self.grid_width_m / 2.0
        g.info.origin.orientation.w = 1.0
        g.data = [-1] * (W * H)
        return g

    # ==================================================================== #
    #  FollowPath action                                                     #
    # ==================================================================== #

    def _send_goal(self, path):
        if not self.enable_follow_path:
            return
        if not path.poses:
            self._cancel_goal()
            return
        if self.require_ctrl_active and not self.ctrl_active:
            self.get_logger().warn(
                f'FollowPath blocked: controller state={self.ctrl_state_label}',
                throttle_duration_sec=2.0)
            return
        if self.required_tf_frame:
            stamp = Time.from_msg(path.header.stamp)
            if not (self.tf_buffer.can_transform(
                        self.required_tf_frame, path.header.frame_id,
                        stamp, timeout=Duration(seconds=0.05)) or
                    self.tf_buffer.can_transform(
                        self.required_tf_frame, path.header.frame_id,
                        Time(), timeout=Duration(seconds=0.1))):
                self.get_logger().warn(
                    f'FollowPath blocked: missing TF '
                    f'{self.required_tf_frame}<-{path.header.frame_id}',
                    throttle_duration_sec=1.0)
                return
        now = self.get_clock().now()
        if self.last_goal_time and \
                (now - self.last_goal_time).nanoseconds < self.min_goal_period_sec * 1e9:
            return
        if self.goal_pending or self.goal_active:
            return
        if not self.fp_client.server_is_ready():
            self.get_logger().warn('FollowPath server not ready.', throttle_duration_sec=2.0)
            return

        goal = FollowPath.Goal()
        goal.path            = path
        goal.controller_id   = self.controller_id
        goal.goal_checker_id = self.goal_checker_id
        self.get_logger().info(f'Sending FollowPath goal: {len(path.poses)} poses',
                               throttle_duration_sec=1.0)
        self.goal_pending   = True
        self.last_goal_time = now
        self.fp_client.send_goal_async(goal).add_done_callback(self._on_goal_response)

    def _on_goal_response(self, future):
        self.goal_pending = False
        try:
            handle = future.result()
        except Exception as e:
            self.get_logger().error(f'FollowPath goal error: {e}')
            return
        if not handle.accepted:
            self.get_logger().warn('FollowPath goal rejected.')
            return
        self.current_goal_handle = handle
        self.goal_active = True
        self.get_logger().info('FollowPath goal accepted.', throttle_duration_sec=1.0)
        handle.get_result_async().add_done_callback(self._on_goal_result)

    def _on_goal_result(self, future):
        self.goal_active         = False
        self.current_goal_handle = None
        try:
            r = future.result()
            self.get_logger().info(f'FollowPath finished, status={r.status}',
                                   throttle_duration_sec=1.0)
        except Exception as e:
            self.get_logger().warn(f'FollowPath result error: {e}')

    def _cancel_goal(self):
        if self.goal_active and self.current_goal_handle:
            self.get_logger().info('Cancelling FollowPath goal.', throttle_duration_sec=1.0)
            fut = self.current_goal_handle.cancel_goal_async()
            fut.add_done_callback(lambda _: (
                setattr(self, 'goal_active', False),
                setattr(self, 'current_goal_handle', None)))

    # ==================================================================== #
    #  Controller lifecycle polling                                          #
    # ==================================================================== #

    def _poll_controller_state(self):
        if self._ctrl_req_pending or not self.ctrl_client.service_is_ready():
            return
        self._ctrl_req_pending = True
        self.ctrl_client.call_async(GetState.Request()).add_done_callback(
            self._on_controller_state)

    def _on_controller_state(self, future):
        self._ctrl_req_pending = False
        try:
            r = future.result()
            self.ctrl_state_label = r.current_state.label
            self.ctrl_active      = (self.ctrl_state_label == 'active')
        except Exception:
            self.ctrl_active      = False
            self.ctrl_state_label = 'unknown'

    # ==================================================================== #
    #  Helpers                                                               #
    # ==================================================================== #

    @staticmethod
    def _interp_y(pts, x):
        """Linear interpolate/extrapolate y at x from a sorted (x,y) list."""
        if len(pts) < 2:
            return None
        if x <= pts[0][0]:
            return pts[0][1]
        if x >= pts[-1][0]:
            return pts[-1][1]
        for (x1, y1), (x2, y2) in zip(pts, pts[1:]):
            if x1 <= x <= x2:
                t = (x - x1) / (x2 - x1) if (x2 - x1) > 1e-6 else 0.0
                return y1 + t * (y2 - y1)
        return None

    def _publish_overlay(self, idx, bgr, hough, left_px, right_px, rgb_msg):
        overlay = bgr.copy()
        if hough is not None:
            for seg in hough:
                cv2.line(overlay,
                         (seg[0][0], seg[0][1]), (seg[0][2], seg[0][3]),
                         (0, 255, 255), 1)
        if left_px:
            cv2.line(overlay, left_px[0],  left_px[1],  (255, 0, 0), 4)
        if right_px:
            cv2.line(overlay, right_px[0], right_px[1], (0, 0, 255), 4)
        label = f'cam={idx}  L={left_px is not None}  R={right_px is not None}'
        cv2.putText(overlay, label, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        try:
            msg = self.bridge.cv2_to_imgmsg(overlay, 'bgr8')
            msg.header = rgb_msg.header
            self.overlay_pubs[idx].publish(msg)
        except Exception as e:
            self.get_logger().warn(f'Overlay publish failed: {e}', throttle_duration_sec=1.0)

    def _watchdog(self):
        if not self._got_frame:
            self.get_logger().warn(
                'No synced RGB+depth frames received. '
                'Check camera/depth topics and ApproximateTimeSynchronizer slop.',
                throttle_duration_sec=5.0)
        self._got_frame = False


# ======================================================================== #

def main(args=None):
    rclpy.init(args=args)
    rclpy.spin(LaneDetectionNode())
    rclpy.shutdown()


if __name__ == '__main__':
    main()