"""
igvc_localization.py

Owns the map → odom transform for the entire IGVC stack.

GPS mode  (gps_enabled: true)
    Monitors /gps/fix health.  While GPS is healthy, robot_localization's
    navsat_transform EKF publishes map→odom and this node stays silent.
    On GPS loss this node snapshots the last good transform and re-broadcasts
    it at publish_rate_hz so Nav2's TF chain never breaks.  On recovery it
    hands control back silently.

Sim mode  (gps_enabled: false)
    No GPS topic is subscribed.  An identity map→odom is seeded immediately
    so Nav2 has a valid TF chain from tick zero.  The node publishes it
    continuously for the lifetime of the process — no health checks, no
    state transitions.

In both modes /localization_status (std_msgs/String) reports the current
state: 'gps' | 'dead_reckoning' | 'sim'.  Other nodes can gate behaviour
on this topic without needing to know the gps_enabled parameter themselves.

Parameters
    gps_enabled           bool    true
    gps_timeout_sec       float   2.0
    gps_min_status        int     0       NavSatFix STATUS_FIX
    publish_rate_hz       float   50.0
    drift_warn_dist_m     float   5.0
    map_frame             str     map
    odom_frame            str     odom
    gps_topic             str     /gps/fix
    odom_topic            str     /odom

Publications
    /tf                       TransformStamped    map→odom  (fallback / sim only)
    /localization_status      std_msgs/String     current mode label
"""

from __future__ import annotations

import math

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.time import Time
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import NavSatFix
from std_msgs.msg import String
from tf2_ros import Buffer, TransformBroadcaster, TransformListener


# ── Constants ─────────────────────────────────────────────────────────────────

_STATUS_GPS  = 'gps'
_STATUS_DR   = 'dead_reckoning'
_STATUS_SIM  = 'sim'
_STATUS_INIT = 'initializing'


# ── Node ──────────────────────────────────────────────────────────────────────

class IGVCLocalizationNode(Node):

    def __init__(self) -> None:
        super().__init__('igvc_localization')

        # ── Parameters ────────────────────────────────────────────────────
        self._declare_params()
        gps_en      = self._p('gps_enabled',        True)
        timeout     = self._p('gps_timeout_sec',     2.0)
        min_status  = self._p('gps_min_status',      0)
        rate_hz     = self._p('publish_rate_hz',    50.0)
        warn_dist   = self._p('drift_warn_dist_m',   5.0)
        map_frame   = self._p('map_frame',          'map')
        odom_frame  = self._p('odom_frame',         'odom')
        gps_topic   = self._p('gps_topic',          '/gps/fix')
        odom_topic  = self._p('odom_topic',         '/odom')

        self._map_frame  = map_frame
        self._odom_frame = odom_frame
        self._gps_enabled   = gps_en
        self._gps_timeout   = timeout
        self._gps_min_status = min_status
        self._warn_dist     = warn_dist

        # ── Internal state ────────────────────────────────────────────────
        self._status: str               = _STATUS_INIT
        self._last_gps_stamp: Time | None = None
        self._snapshot: TransformStamped | None = None  # last good map→odom
        self._dr_distance: float        = 0.0
        self._prev_odom_xy: tuple[float, float] | None = None

        # ── TF ────────────────────────────────────────────────────────────
        self._tf_buf       = Buffer()
        self._tf_listener  = TransformListener(self._tf_buf, self)
        self._tf_broadcast = TransformBroadcaster(self)

        # ── Publishers ────────────────────────────────────────────────────
        self._status_pub = self.create_publisher(String, '/localization_status', 10)

        # ── Subscribers ───────────────────────────────────────────────────
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST, depth=5)

        self.create_subscription(Odometry, odom_topic, self._on_odom, 10)

        if gps_en:
            self.create_subscription(NavSatFix, gps_topic,
                                     self._on_gps, sensor_qos)
            self.get_logger().info(f'Localization: GPS mode — monitoring {gps_topic}')
        else:
            self._status = _STATUS_SIM
            self._seed_identity()
            self.get_logger().info(
                'Localization: sim mode — identity map→odom seeded, no GPS checks.')

        self._publish_status()

        # ── Timers ────────────────────────────────────────────────────────
        self.create_timer(1.0 / rate_hz, self._broadcast)
        self.create_timer(1.0,           self._publish_status)

    # ── Parameter helpers ─────────────────────────────────────────────────

    def _declare_params(self) -> None:
        for name, default in [
            ('gps_enabled',       True),
            ('gps_timeout_sec',   2.0),
            ('gps_min_status',    0),
            ('publish_rate_hz',  50.0),
            ('drift_warn_dist_m', 5.0),
            ('map_frame',        'map'),
            ('odom_frame',       'odom'),
            ('gps_topic',        '/gps/fix'),
            ('odom_topic',       '/odom'),
        ]:
            self.declare_parameter(name, default)

    def _p(self, name: str, _default):
        return self.get_parameter(name).value

    # ── GPS callback ──────────────────────────────────────────────────────

    def _on_gps(self, msg: NavSatFix) -> None:
        if msg.status.status < self._gps_min_status:
            return
        self._last_gps_stamp = self.get_clock().now()
        self._refresh_snapshot()

        if self._status != _STATUS_GPS:
            prev = self._status
            self._status    = _STATUS_GPS
            self._dr_distance = 0.0
            if prev == _STATUS_DR:
                self.get_logger().info(
                    f'GPS recovered after {self._dr_distance:.1f} m — '
                    'handing map→odom back to robot_localization.')

    # ── Odometry callback ─────────────────────────────────────────────────

    def _on_odom(self, msg: Odometry) -> None:
        xy = (msg.pose.pose.position.x, msg.pose.pose.position.y)
        if self._prev_odom_xy is not None and self._status == _STATUS_DR:
            self._dr_distance += math.hypot(
                xy[0] - self._prev_odom_xy[0],
                xy[1] - self._prev_odom_xy[1])
        self._prev_odom_xy = xy

    # ── Broadcast timer ───────────────────────────────────────────────────

    def _broadcast(self) -> None:
        now = self.get_clock().now()

        if self._status == _STATUS_SIM:
            self._send_snapshot(now)
            return

        if self._status == _STATUS_GPS:
            # robot_localization owns the transform — stay silent
            return

        # INIT or DR: check GPS age
        gps_age = (
            (now - self._last_gps_stamp).nanoseconds / 1e9
            if self._last_gps_stamp else float('inf'))

        if gps_age < self._gps_timeout:
            # GPS just came back — _on_gps will flip status
            return

        if self._status == _STATUS_INIT:
            if self._snapshot is None:
                self.get_logger().warn(
                    'Waiting for first GPS fix to initialise map→odom.',
                    throttle_duration_sec=5.0)
                return

        if self._status == _STATUS_GPS:
            self.get_logger().warn(
                f'GPS lost ({gps_age:.1f} s ago) — switching to dead reckoning.')
            self._status        = _STATUS_DR
            self._dr_distance   = 0.0
            self._prev_odom_xy  = None

        self._send_snapshot(now)

        if self._dr_distance > self._warn_dist:
            self.get_logger().warn(
                f'Dead reckoning {self._dr_distance:.1f} m, '
                f'est. lateral error ~{self._dr_distance * 0.02:.2f} m.',
                throttle_duration_sec=5.0)

    # ── Status publisher ──────────────────────────────────────────────────

    def _publish_status(self) -> None:
        msg = String(data=self._status)
        self._status_pub.publish(msg)
        self.get_logger().debug(
            f'[{self._status}] dr_dist={self._dr_distance:.1f} m',
            throttle_duration_sec=2.0)

    # ── Helpers ───────────────────────────────────────────────────────────

    def _seed_identity(self) -> None:
        tf = TransformStamped()
        tf.header.frame_id      = self._map_frame
        tf.child_frame_id       = self._odom_frame
        tf.transform.rotation.w = 1.0
        self._snapshot = tf

    def _refresh_snapshot(self) -> None:
        try:
            tf = self._tf_buf.lookup_transform(
                self._map_frame, self._odom_frame,
                Time(), timeout=Duration(seconds=0.05))
            self._snapshot = tf
            if self._status == _STATUS_INIT:
                self._status = _STATUS_GPS
                self.get_logger().info('First map→odom captured — localization initialised.')
        except Exception:
            pass

    def _send_snapshot(self, now: Time) -> None:
        if self._snapshot is None:
            return
        tf = TransformStamped()
        tf.header.stamp    = now.to_msg()
        tf.header.frame_id = self._map_frame
        tf.child_frame_id  = self._odom_frame
        tf.transform       = self._snapshot.transform
        self._tf_broadcast.sendTransform(tf)


# ── Entry point ───────────────────────────────────────────────────────────────

def main(args=None) -> None:
    rclpy.init(args=args)
    rclpy.spin(IGVCLocalizationNode())
    rclpy.shutdown()


if __name__ == '__main__':
    main()