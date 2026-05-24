"""
brake_control.py
================

Applies or releases the electro-mechanical brakes via GPIO and
co-ordinates two safety actions to prevent the robot moving while
the brakes are engaged:

  1. Calls ``/controller_manager/switch_controller`` to deactivate the
     ``diff_drive_controller``.  This triggers
     ``ODriveHardwareInterface::on_deactivate()`` which sends
     ``AXIS_STATE_IDLE`` to every ODrive over CAN — actively dropping
     motor torque before the brake pads clamp.

  2. Calls ``/igvc_navigator/set_paused`` (``std_srvs/SetBool``) so the
     lane navigator does not issue new Nav2 goals while the brakes are on.

On release (A button) the sequence is reversed: GPIO drops first, then
the diff_drive_controller is re-activated and the navigator unpaused.

Button mapping  (sensor_msgs/Joy ``buttons`` array)
----------------------------------------------------
``B`` button (index 1 — standard Xbox / Logitech layout)
    → brakes APPLIED   (GPIO HIGH, controller idle, navigator paused)
``A`` button (index 0)
    → brakes RELEASED  (GPIO LOW, controller active, navigator resumed)

Both button indices are ROS 2 parameters for controller remapping.

Topics / services
-----------------
``/joy``                                   ``sensor_msgs/Joy`` (in)
``/controller_manager/switch_controller``  SwitchController    (client)
``/igvc_navigator/set_paused``             ``std_srvs/SetBool`` (client)

GPIO (Jetson, BOARD pin numbering)
-----------------------------------
Pin 22 is driven HIGH to apply brakes and LOW to release.
Initialised LOW on startup.
"""

from __future__ import annotations

import threading

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from controller_manager_msgs.srv import SwitchController
from sensor_msgs.msg import Joy
from std_srvs.srv import SetBool

try:
    import Jetson.GPIO as GPIO
    _GPIO_AVAILABLE = True
except ImportError:
    _GPIO_AVAILABLE = False

# Physical BOARD pin number for the brake solenoid output.
_BRAKE_PIN: int = 22

# ros2_control diff_drive controller name (must match controllers.yaml).
_DRIVE_CONTROLLER = 'diff_drive_controller'


class BrakeControlNode(Node):

    def __init__(self) -> None:
        super().__init__('brake_control_node')

        # ── Parameters ────────────────────────────────────────────────
        p = self.declare_parameter
        self._btn_apply   = int(p('button_apply_brakes',   1).value)  # B
        self._btn_release = int(p('button_release_brakes', 0).value)  # A
        self._joy_topic   = p('joy_topic', '/joy').value
        self._nav_srv     = p('navigator_pause_service',
                               '/igvc_navigator/set_paused').value
        self._ctrl_srv    = p('switch_controller_service',
                               '/controller_manager/switch_controller').value
        self._srv_timeout = float(p('service_timeout_sec', 2.0).value)

        # ── GPIO ──────────────────────────────────────────────────────
        if _GPIO_AVAILABLE:
            GPIO.setmode(GPIO.BOARD)
            GPIO.setup(_BRAKE_PIN, GPIO.OUT, initial=GPIO.LOW)
            self.get_logger().info(
                f'brake_control: GPIO BOARD pin {_BRAKE_PIN} initialised LOW '
                '(brakes released)')
        else:
            self.get_logger().warn(
                'brake_control: Jetson.GPIO not available — mock mode')

        self._brakes_applied = False
        self._lock = threading.Lock()  # guards _brakes_applied across threads

        # ── Service clients ───────────────────────────────────────────
        self._ctrl_client = self.create_client(
            SwitchController, self._ctrl_srv)
        self._nav_client = self.create_client(
            SetBool, self._nav_srv)

        # ── Joy subscription ──────────────────────────────────────────
        self.create_subscription(
            Joy, self._joy_topic, self._on_joy, qos_profile_sensor_data)

        self.get_logger().info(
            f'brake_control_node ready — '
            f'apply=buttons[{self._btn_apply}] '
            f'release=buttons[{self._btn_release}]')

    # ── Joystick callback (spins off a worker thread) ─────────────────────

    def _on_joy(self, msg: Joy) -> None:
        n = len(msg.buttons)
        apply   = (self._btn_apply   < n and bool(msg.buttons[self._btn_apply]))
        release = (self._btn_release < n and bool(msg.buttons[self._btn_release]))

        with self._lock:
            if apply and not self._brakes_applied:
                # Mark immediately so repeat button presses don't double-fire.
                self._brakes_applied = True
                threading.Thread(
                    target=self._apply_sequence, daemon=True).start()
            elif release and self._brakes_applied:
                self._brakes_applied = False
                threading.Thread(
                    target=self._release_sequence, daemon=True).start()

    # ── Apply sequence ────────────────────────────────────────────────────
    # Order: pause navigator → idle motor controllers → set GPIO high.
    # Each step is attempted even if the previous one fails so GPIO always
    # follows through.

    def _apply_sequence(self) -> None:
        self.get_logger().info('brake_control: APPLYING brakes — '
                               'pausing navigator and idling motor controllers')

        # 1. Pause the navigator so it stops issuing Nav2 goals.
        self._call_nav_pause(pause=True)

        # 2. Deactivate diff_drive_controller → triggers AXIS_STATE_IDLE on ODrives.
        self._call_switch_controller(activate=False)

        # 3. Engage the physical brake.
        self._write_gpio(True)

        self.get_logger().info('brake_control: brakes APPLIED '
                               f'(GPIO pin {_BRAKE_PIN} HIGH)')

    # ── Release sequence ──────────────────────────────────────────────────
    # Order: release GPIO → re-activate motor controllers → unpause navigator.

    def _release_sequence(self) -> None:
        self.get_logger().info('brake_control: RELEASING brakes — '
                               'enabling motor controllers and resuming navigator')

        # 1. Release the physical brake first so the ODrive isn't fighting
        #    the brake pad as it re-activates.
        self._write_gpio(False)

        # 2. Re-activate diff_drive_controller → ODrive enters closed-loop control.
        self._call_switch_controller(activate=True)

        # 3. Resume the navigator.
        self._call_nav_pause(pause=False)

        self.get_logger().info('brake_control: brakes RELEASED '
                               f'(GPIO pin {_BRAKE_PIN} LOW)')

    # ── Helpers ───────────────────────────────────────────────────────────

    def _write_gpio(self, high: bool) -> None:
        if _GPIO_AVAILABLE:
            GPIO.output(_BRAKE_PIN, GPIO.HIGH if high else GPIO.LOW)

    def _call_nav_pause(self, pause: bool) -> None:
        if not self._nav_client.wait_for_service(timeout_sec=self._srv_timeout):
            self.get_logger().warn(
                f'brake_control: {self._nav_srv} not available — skipping')
            return
        req = SetBool.Request()
        req.data = pause
        future = self._nav_client.call_async(req)
        rclpy.spin_until_future_complete(
            self, future, timeout_sec=self._srv_timeout)
        if future.done() and future.result() is not None:
            self.get_logger().debug(
                f'brake_control: navigator {"paused" if pause else "resumed"}: '
                f'{future.result().message}')
        else:
            self.get_logger().warn(
                f'brake_control: navigator pause call timed out '
                f'(pause={pause})')

    def _call_switch_controller(self, activate: bool) -> None:
        if not self._ctrl_client.wait_for_service(timeout_sec=self._srv_timeout):
            self.get_logger().warn(
                f'brake_control: {self._ctrl_srv} not available — skipping')
            return
        req = SwitchController.Request()
        if activate:
            req.activate_controllers   = [_DRIVE_CONTROLLER]
            req.deactivate_controllers = []
        else:
            req.activate_controllers   = []
            req.deactivate_controllers = [_DRIVE_CONTROLLER]
        req.strictness = SwitchController.Request.BEST_EFFORT
        req.activate_asap = True
        future = self._ctrl_client.call_async(req)
        rclpy.spin_until_future_complete(
            self, future, timeout_sec=self._srv_timeout)
        if future.done() and future.result() is not None:
            action = 'activated' if activate else 'deactivated'
            ok     = future.result().ok
            self.get_logger().info(
                f'brake_control: {_DRIVE_CONTROLLER} {action} '
                f'(ok={ok})')
        else:
            self.get_logger().warn(
                f'brake_control: switch_controller call timed out '
                f'(activate={activate})')

    # ── Shutdown ──────────────────────────────────────────────────────────

    def destroy_node(self) -> None:
        # Safety: release brakes on shutdown so the robot is not locked.
        if self._brakes_applied:
            self.get_logger().warn(
                'brake_control: releasing brakes on shutdown (safety)')
            self._write_gpio(False)
            self._call_switch_controller(activate=True)
            self._call_nav_pause(pause=False)
        if _GPIO_AVAILABLE:
            GPIO.cleanup(_BRAKE_PIN)
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


if __name__ == '__main__':
    main()
