#!/usr/bin/env python3

import math
import time
from dataclasses import dataclass

import rclpy
from geometry_msgs.msg import QuaternionStamped
from rclpy.node import Node
from sensor_msgs.msg import Imu
from std_msgs.msg import Float64


class I2CDependencyError(RuntimeError):
    pass


def _load_smbus_class():
    try:
        from smbus2 import SMBus

        return SMBus
    except ImportError:
        try:
            from smbus import SMBus

            return SMBus
        except ImportError as exc:
            raise I2CDependencyError(
                'Install python3-smbus or smbus2 to read the L3GD20 over I2C.'
            ) from exc


def _parse_int(value) -> int:
    if isinstance(value, str):
        return int(value, 0)
    return int(value)


def _wrap_pi(angle_rad: float) -> float:
    return math.atan2(math.sin(angle_rad), math.cos(angle_rad))


def _wrap_360(angle_rad: float) -> float:
    return math.degrees(angle_rad) % 360.0


def _signed_16(lsb: int, msb: int) -> int:
    value = (msb << 8) | lsb
    if value & 0x8000:
        value -= 0x10000
    return value


@dataclass(frozen=True)
class GyroSample:
    x: float
    y: float
    z: float


class L3GD20Driver:
    WHO_AM_I = 0x0F
    CTRL_REG1 = 0x20
    CTRL_REG4 = 0x23
    CTRL_REG5 = 0x24
    OUT_X_L = 0x28
    AUTO_INCREMENT = 0x80

    WHO_AM_I_VALUES = {0xD4, 0xD7}
    SCALE_MDPS_PER_LSB = {
        250: 8.75,
        500: 17.5,
        2000: 70.0,
    }
    FS_BITS = {
        250: 0x00,
        500: 0x10,
        2000: 0x20,
    }

    def __init__(self, bus_number: int, address: int, full_scale_dps: int):
        bus_class = _load_smbus_class()
        self._bus = bus_class(bus_number)
        self._address = address
        self._full_scale_dps = self._coerce_full_scale(full_scale_dps)
        self._scale_rad_s = (
            self.SCALE_MDPS_PER_LSB[self._full_scale_dps] * 0.001 * math.pi / 180.0
        )

        who_am_i = self._bus.read_byte_data(self._address, self.WHO_AM_I)
        if who_am_i not in self.WHO_AM_I_VALUES:
            self.close()
            expected = ', '.join(f'0x{value:02x}' for value in sorted(self.WHO_AM_I_VALUES))
            raise RuntimeError(
                f'L3GD20 WHO_AM_I mismatch at 0x{self._address:02x}: '
                f'got 0x{who_am_i:02x}, expected one of {expected}'
            )

        self.configure()

    @classmethod
    def _coerce_full_scale(cls, full_scale_dps: int) -> int:
        full_scale_dps = int(full_scale_dps)
        if full_scale_dps in cls.FS_BITS:
            return full_scale_dps
        raise ValueError('full_scale_dps must be one of 250, 500, or 2000')

    @property
    def full_scale_dps(self) -> int:
        return self._full_scale_dps

    def configure(self) -> None:
        self._bus.write_byte_data(self._address, self.CTRL_REG1, 0x0F)
        self._bus.write_byte_data(
            self._address,
            self.CTRL_REG4,
            0x80 | self.FS_BITS[self._full_scale_dps],
        )
        self._bus.write_byte_data(self._address, self.CTRL_REG5, 0x00)
        time.sleep(0.1)

    def read_angular_velocity(self) -> GyroSample:
        data = self._bus.read_i2c_block_data(
            self._address,
            self.OUT_X_L | self.AUTO_INCREMENT,
            6,
        )
        return GyroSample(
            x=_signed_16(data[0], data[1]) * self._scale_rad_s,
            y=_signed_16(data[2], data[3]) * self._scale_rad_s,
            z=_signed_16(data[4], data[5]) * self._scale_rad_s,
        )

    def close(self) -> None:
        close = getattr(self._bus, 'close', None)
        if close is not None:
            close()


class L3GD20HeadingNode(Node):
    def __init__(self):
        super().__init__('l3gd20_heading_node')

        self.declare_parameter('i2c_bus', 0)
        self.declare_parameter('i2c_address', 107)
        self.declare_parameter('full_scale_dps', 250)
        self.declare_parameter('sample_rate_hz', 95.0)
        self.declare_parameter('frame_id', 'l3gd20_link')
        self.declare_parameter('world_frame_id', 'odom')
        self.declare_parameter('heading_topic', '/navheading')
        self.declare_parameter('heading_degrees_topic', '/navheading_deg')
        self.declare_parameter('heading_quaternion_topic', '/l3gd20/heading')
        self.declare_parameter('imu_topic', '/l3gd20/imu')
        self.declare_parameter('publish_heading_degrees', True)
        self.declare_parameter('publish_imu', True)
        self.declare_parameter('publish_heading_quaternion', True)
        self.declare_parameter('initial_heading_rad', 0.0)
        self.declare_parameter('yaw_axis', 'z')
        self.declare_parameter('yaw_sign', 1.0)
        self.declare_parameter('deadband_dps', 0.03)
        self.declare_parameter('calibration_samples', 250)
        self.declare_parameter('reconnect_period_sec', 2.0)
        self.declare_parameter('max_dt_sec', 0.25)
        self.declare_parameter('orientation_yaw_variance', 0.05)
        self.declare_parameter('angular_velocity_variance', 0.01)

        self._i2c_bus = _parse_int(self.get_parameter('i2c_bus').value)
        self._i2c_address = _parse_int(self.get_parameter('i2c_address').value)
        self._full_scale_dps = _parse_int(self.get_parameter('full_scale_dps').value)
        self._sample_rate_hz = max(1.0, float(self.get_parameter('sample_rate_hz').value))
        self._frame_id = str(self.get_parameter('frame_id').value)
        self._world_frame_id = str(self.get_parameter('world_frame_id').value)
        self._yaw_axis = str(self.get_parameter('yaw_axis').value).lower()
        self._yaw_sign = float(self.get_parameter('yaw_sign').value)
        self._deadband_rad_s = math.radians(float(self.get_parameter('deadband_dps').value))
        self._calibration_target = max(
            0,
            _parse_int(self.get_parameter('calibration_samples').value),
        )
        self._reconnect_period_sec = max(
            0.1,
            float(self.get_parameter('reconnect_period_sec').value),
        )
        self._max_dt_sec = max(0.01, float(self.get_parameter('max_dt_sec').value))
        self._orientation_yaw_variance = max(
            0.0,
            float(self.get_parameter('orientation_yaw_variance').value),
        )
        self._angular_velocity_variance = max(
            0.0,
            float(self.get_parameter('angular_velocity_variance').value),
        )

        if self._yaw_axis not in ('x', 'y', 'z'):
            raise ValueError('yaw_axis must be x, y, or z')

        self._heading_rad = _wrap_pi(float(self.get_parameter('initial_heading_rad').value))
        self._bias = GyroSample(0.0, 0.0, 0.0)
        self._calibration_sum = GyroSample(0.0, 0.0, 0.0)
        self._calibration_count = 0
        self._last_stamp = None
        self._driver = None
        self._next_connect_time = 0.0

        self._heading_pub = self.create_publisher(
            Float64,
            str(self.get_parameter('heading_topic').value),
            10,
        )
        self._heading_deg_pub = None
        if bool(self.get_parameter('publish_heading_degrees').value):
            self._heading_deg_pub = self.create_publisher(
                Float64,
                str(self.get_parameter('heading_degrees_topic').value),
                10,
            )

        self._heading_quat_pub = None
        if bool(self.get_parameter('publish_heading_quaternion').value):
            self._heading_quat_pub = self.create_publisher(
                QuaternionStamped,
                str(self.get_parameter('heading_quaternion_topic').value),
                10,
            )

        self._imu_pub = None
        if bool(self.get_parameter('publish_imu').value):
            self._imu_pub = self.create_publisher(
                Imu,
                str(self.get_parameter('imu_topic').value),
                10,
            )

        self._connect()
        self._timer = self.create_timer(1.0 / self._sample_rate_hz, self._sample)

    def destroy_node(self):
        self._close_driver()
        return super().destroy_node()

    def _connect(self) -> None:
        try:
            self._driver = L3GD20Driver(
                self._i2c_bus,
                self._i2c_address,
                self._full_scale_dps,
            )
        except Exception as exc:
            self._driver = None
            self._next_connect_time = time.monotonic() + self._reconnect_period_sec
            self.get_logger().warn(
                f'L3GD20 not available on /dev/i2c-{self._i2c_bus} '
                f'address 0x{self._i2c_address:02x} '
                f'(Jetson header pins 27 SDA / 28 SCL): {exc}'
            )
            return

        self._calibration_sum = GyroSample(0.0, 0.0, 0.0)
        self._calibration_count = 0
        self._last_stamp = None
        self.get_logger().info(
            f'L3GD20 connected on /dev/i2c-{self._i2c_bus} '
            f'address 0x{self._i2c_address:02x}; full_scale={self._driver.full_scale_dps} dps; '
            f'calibration_samples={self._calibration_target}'
        )
        if self._calibration_target > 0:
            self.get_logger().info('Keep the robot still while gyro bias calibration runs.')

    def _close_driver(self) -> None:
        if self._driver is not None:
            self._driver.close()
            self._driver = None

    def _sample(self) -> None:
        if self._driver is None:
            if time.monotonic() >= self._next_connect_time:
                self._connect()
            return

        try:
            sample = self._driver.read_angular_velocity()
        except Exception as exc:
            self.get_logger().warn(f'L3GD20 read failed, will reconnect: {exc}')
            self._close_driver()
            self._next_connect_time = time.monotonic() + self._reconnect_period_sec
            return

        stamp = self.get_clock().now()
        if self._calibration_count < self._calibration_target:
            self._accumulate_calibration(sample, stamp)
            return

        corrected = GyroSample(
            sample.x - self._bias.x,
            sample.y - self._bias.y,
            sample.z - self._bias.z,
        )
        yaw_rate = getattr(corrected, self._yaw_axis) * self._yaw_sign
        if abs(yaw_rate) < self._deadband_rad_s:
            yaw_rate = 0.0

        if self._last_stamp is None:
            self._last_stamp = stamp
            self._publish(stamp, corrected)
            return

        dt = (stamp - self._last_stamp).nanoseconds * 1.0e-9
        self._last_stamp = stamp
        if dt <= 0.0:
            return
        if dt > self._max_dt_sec:
            self.get_logger().warn(
                f'Skipping L3GD20 integration step after large dt={dt:.3f}s'
            )
            self._publish(stamp, corrected)
            return

        self._heading_rad = _wrap_pi(self._heading_rad + yaw_rate * dt)
        self._publish(stamp, corrected)

    def _accumulate_calibration(self, sample: GyroSample, stamp) -> None:
        self._calibration_sum = GyroSample(
            self._calibration_sum.x + sample.x,
            self._calibration_sum.y + sample.y,
            self._calibration_sum.z + sample.z,
        )
        self._calibration_count += 1
        if self._calibration_count >= self._calibration_target:
            denominator = float(max(1, self._calibration_count))
            self._bias = GyroSample(
                self._calibration_sum.x / denominator,
                self._calibration_sum.y / denominator,
                self._calibration_sum.z / denominator,
            )
            self._last_stamp = stamp
            self.get_logger().info(
                'L3GD20 gyro bias calibrated '
                f'(rad/s): x={self._bias.x:.6f}, y={self._bias.y:.6f}, z={self._bias.z:.6f}'
            )

    def _publish(self, stamp, corrected: GyroSample) -> None:
        heading_msg = Float64()
        heading_msg.data = self._heading_rad
        self._heading_pub.publish(heading_msg)

        if self._heading_deg_pub is not None:
            heading_deg_msg = Float64()
            heading_deg_msg.data = _wrap_360(self._heading_rad)
            self._heading_deg_pub.publish(heading_deg_msg)

        quaternion = self._yaw_to_quaternion(self._heading_rad)

        if self._heading_quat_pub is not None:
            quat_msg = QuaternionStamped()
            quat_msg.header.stamp = stamp.to_msg()
            quat_msg.header.frame_id = self._world_frame_id
            quat_msg.quaternion.x = quaternion[0]
            quat_msg.quaternion.y = quaternion[1]
            quat_msg.quaternion.z = quaternion[2]
            quat_msg.quaternion.w = quaternion[3]
            self._heading_quat_pub.publish(quat_msg)

        if self._imu_pub is not None:
            imu_msg = Imu()
            imu_msg.header.stamp = stamp.to_msg()
            imu_msg.header.frame_id = self._frame_id
            imu_msg.orientation.x = quaternion[0]
            imu_msg.orientation.y = quaternion[1]
            imu_msg.orientation.z = quaternion[2]
            imu_msg.orientation.w = quaternion[3]
            imu_msg.orientation_covariance = [
                1.0e6,
                0.0,
                0.0,
                0.0,
                1.0e6,
                0.0,
                0.0,
                0.0,
                self._orientation_yaw_variance,
            ]
            imu_msg.angular_velocity.x = corrected.x
            imu_msg.angular_velocity.y = corrected.y
            imu_msg.angular_velocity.z = corrected.z
            imu_msg.angular_velocity_covariance = [
                self._angular_velocity_variance,
                0.0,
                0.0,
                0.0,
                self._angular_velocity_variance,
                0.0,
                0.0,
                0.0,
                self._angular_velocity_variance,
            ]
            imu_msg.linear_acceleration_covariance[0] = -1.0
            self._imu_pub.publish(imu_msg)

    @staticmethod
    def _yaw_to_quaternion(yaw_rad: float) -> tuple[float, float, float, float]:
        half_yaw = yaw_rad * 0.5
        return (0.0, 0.0, math.sin(half_yaw), math.cos(half_yaw))


def main(args=None):
    rclpy.init(args=args)
    node = L3GD20HeadingNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()