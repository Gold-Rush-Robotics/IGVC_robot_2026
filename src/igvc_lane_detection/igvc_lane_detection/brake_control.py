"""
brake_control.py
================

Applies or releases the electro-mechanical brakes via GPIO based on
joystick button presses.

Button mapping (sensor_msgs/Joy, ``buttons`` array)
----------------------------------------------------
``B`` button (index 1 — standard Xbox / Logitech layout)
    → GPIO pins 18 and 22 HIGH  (brakes APPLIED)
``A`` button (index 0)
    → GPIO pins 18 and 22 LOW   (brakes RELEASED)

The button indices are ROS 2 parameters so they can be overridden
for different controllers without recompiling.

Topics
------
``/joy``  ``sensor_msgs/Joy``  (sensor QoS, in)

GPIO
----
Pins 18 and 22 (BOARD numbering) are driven HIGH to apply brakes and
LOW to release.  Both pins are initialised LOW on startup.
"""

from __future__ import annotations

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Joy

try:
    import Jetson.GPIO as GPIO
    _GPIO_AVAILABLE = True
except ImportError:
    _GPIO_AVAILABLE = False

# Physical BOARD pin numbers for the brake solenoid outputs.
_BRAKE_PINS: int = 22


class BrakeControlNode(Node):

    def __init__(self) -> None:
        super().__init__('brake_control_node')

        # ── Parameters ────────────────────────────────────────────────
        p = self.declare_parameter
        self._btn_apply   = int(p('button_apply_brakes',   1).value)  # B
        self._btn_release = int(p('button_release_brakes', 0).value)  # A
        self._joy_topic   = p('joy_topic', '/joy').value

        # ── GPIO setup ────────────────────────────────────────────────
        if _GPIO_AVAILABLE:
            GPIO.setmode(GPIO.BOARD)
            GPIO.setup(_BRAKE_PINS, GPIO.OUT, initial=GPIO.LOW)
            self.get_logger().info(
                f'brake_control: GPIO BOARD pin {_BRAKE_PINS} initialised LOW '
                '(brakes released)')
        else:
            self.get_logger().warn(
                'brake_control: Jetson.GPIO not available — '
                'running in mock mode (no hardware output)')

        self._brakes_applied = False

        # ── Subscription ──────────────────────────────────────────────
        self.create_subscription(
            Joy, self._joy_topic, self._on_joy, qos_profile_sensor_data)

        self.get_logger().info(
            f'brake_control_node ready.  '
            f'joy_topic={self._joy_topic}  '
            f'apply=buttons[{self._btn_apply}]  '
            f'release=buttons[{self._btn_release}]')

    # ── Callback ──────────────────────────────────────────────────────────

    def _on_joy(self, msg: Joy) -> None:
        n = len(msg.buttons)

        apply   = (self._btn_apply   < n and bool(msg.buttons[self._btn_apply]))
        release = (self._btn_release < n and bool(msg.buttons[self._btn_release]))

        if apply and not self._brakes_applied:
            self._set_brakes(True)
        elif release and self._brakes_applied:
            self._set_brakes(False)

    # ── GPIO helper ───────────────────────────────────────────────────────

    def _set_brakes(self, apply: bool) -> None:
        self._brakes_applied = apply
        level = True if apply else False
        if _GPIO_AVAILABLE:
            GPIO.output(_BRAKE_PINS, GPIO.HIGH if apply else GPIO.LOW)
        self.get_logger().info(
            f'brake_control: brakes {"APPLIED" if apply else "RELEASED"} '
            f'(pins {_BRAKE_PINS} → {"HIGH" if apply else "LOW"})')

    # ── Shutdown ──────────────────────────────────────────────────────────

    def destroy_node(self) -> None:
        # Safety: always release brakes on shutdown so the robot is not
        # left in a locked state if the node crashes.
        if self._brakes_applied:
            self.get_logger().warn(
                'brake_control: releasing brakes on shutdown (safety measure)')
            self._set_brakes(False)
        if _GPIO_AVAILABLE:
            GPIO.cleanup(_BRAKE_PINS)
        super().destroy_node()


def main(argv=None) -> None:
    rclpy.init(args=argv)
    node = BrakeControlNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
