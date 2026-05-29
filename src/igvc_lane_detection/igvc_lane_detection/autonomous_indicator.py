"""
autonomous_indicator.py
=======================

Indicates autonomous mode by blinking the LED on GPIO pin 13 at 2 Hz.
When autonomous mode is off (or the node exits) the LED is held ON so
the operator can always see the board is live.

Service
-------
``~/set_autonomous``  ``std_srvs/SetBool``
    data=true  → enter autonomous mode (LED blinks 2 Hz)
    data=false → exit autonomous mode  (LED solid ON)

The node is designed to launch alongside the motor controllers
(``motor_controllers.launch.py``) and is always running; Nav2 /
the navigator call the service to reflect their current mode.
"""

from __future__ import annotations

import rclpy
from rclpy.node import Node
from std_srvs.srv import SetBool

try:
    import Jetson.GPIO as GPIO
    _GPIO_AVAILABLE = True
except ImportError:
    _GPIO_AVAILABLE = False

# Physical BOARD pin number for the status LED.
_LED_PIN = 13


class AutonomousIndicatorNode(Node):

    def __init__(self) -> None:
        super().__init__('autonomous_indicator_node')

        self._autonomous = True
        self._led_state  = True   # current LED output level

        # ── GPIO setup ────────────────────────────────────────────────
        if _GPIO_AVAILABLE:
            GPIO.setmode(GPIO.BOARD)
            GPIO.setup(_LED_PIN, GPIO.OUT, initial=GPIO.HIGH)
            self.get_logger().info(
                f'autonomous_indicator: GPIO BOARD pin {_LED_PIN} initialised (LED ON)')
        else:
            self.get_logger().warn(
                'autonomous_indicator: Jetson.GPIO not available — '
                'running in mock mode (no hardware output)')

        self._led_state = True   # we start with LED ON

        # ── Service ───────────────────────────────────────────────────
        self._srv = self.create_service(
            SetBool, '~/set_autonomous', self._on_set_autonomous)

        # ── 2 Hz blink timer (always running; only acts in auto mode) ──
        self._timer = self.create_timer(0.25, self._tick)  # 4× per blink cycle

        # Wall-clock tick counter drives the 2 Hz pattern.
        self._tick_count = 0

        self.get_logger().info(
            'autonomous_indicator_node ready.  '
            'Call ~/set_autonomous (std_srvs/SetBool) to toggle.')

    # ── Service callback ──────────────────────────────────────────────────

    def _on_set_autonomous(self, req: SetBool.Request,
                           resp: SetBool.Response) -> SetBool.Response:
        self._autonomous = req.data
        if not self._autonomous:
            self._set_led(True)   # solid ON when not in auto mode
        resp.success = True
        resp.message = ('autonomous mode ON' if self._autonomous
                        else 'autonomous mode OFF — LED solid ON')
        self.get_logger().info(f'autonomous_indicator: {resp.message}')
        return resp

    # ── Timer callback ────────────────────────────────────────────────────

    def _tick(self) -> None:
        if not self._autonomous:
            return
        # 2 blinks per second = 500 ms period.
        # Timer fires every 250 ms → toggle every tick to get 2 Hz.
        self._tick_count += 1
        self._set_led(bool(self._tick_count % 2))

    # ── LED helper ────────────────────────────────────────────────────────

    def _set_led(self, on: bool) -> None:
        if self._led_state == on:
            return
        self._led_state = on
        if _GPIO_AVAILABLE:
            GPIO.output(_LED_PIN, GPIO.HIGH if on else GPIO.LOW)

    # ── Shutdown ──────────────────────────────────────────────────────────

    def destroy_node(self) -> None:
        # Ensure LED is left ON so the operator sees the board is still alive.
        self._set_led(True)
        if _GPIO_AVAILABLE:
            GPIO.output(_LED_PIN, GPIO.HIGH)
            GPIO.cleanup(_LED_PIN)
        self.get_logger().info('autonomous_indicator: LED held ON, GPIO cleaned up.')
        super().destroy_node()


def main(argv=None) -> None:
    rclpy.init(args=argv)
    node = AutonomousIndicatorNode()
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
