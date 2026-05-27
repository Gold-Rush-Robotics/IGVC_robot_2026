"""ROS2 node and Qt thread bridge for the IGVC task GUI."""

from __future__ import annotations

import json
import sys
from typing import Any

import cv2
import numpy as np
import rclpy
from ament_index_python.packages import get_package_share_directory
from cv_bridge import CvBridge
from PyQt5.QtCore import QObject, QThread, pyqtSignal
from rcl_interfaces.msg import Parameter as RclParameter
from rcl_interfaces.msg import ParameterType, ParameterValue
from rcl_interfaces.srv import SetParameters
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from std_srvs.srv import Trigger
from yolo_msgs.msg import DetectionArray

from igvc_task_gui.detection_overlay import draw_detections


# ---------------------------------------------------------------------------
# ROS2 node
# ---------------------------------------------------------------------------

class TaskGuiNode(Node, QObject):
    """ROS2 node that bridges the runner topics/services to Qt signals.

    All signals are emitted from the ROS2 executor thread; Qt automatically
    queues them to the GUI thread because the connections are cross-thread.
    """

    # Signals
    status_updated = pyqtSignal(dict)       # parsed /igvc/task_status JSON
    maneuver_updated = pyqtSignal(dict)     # parsed /igvc/maneuver_command JSON
    prediction_updated = pyqtSignal(dict)  # parsed /igvc/yolo_prediction JSON
    image_updated = pyqtSignal(np.ndarray) # BGR image with overlays
    service_feedback = pyqtSignal(str)     # brief human-readable result

    def __init__(
        self,
        task_runner_node: str = 'igvc_task_runner',
        camera_topic: str = '/front_zed_camera_x/rgb/image_raw',
        detection_topic: str = '/detections',
    ) -> None:
        Node.__init__(self, 'igvc_task_gui')
        QObject.__init__(self)

        self._runner_node = task_runner_node
        self._bridge = CvBridge()
        self._latest_detections: list = []

        # ---- subscriptions -----------------------------------------------
        self.create_subscription(
            String, '/igvc/task_status', self._on_task_status, 10)
        self.create_subscription(
            String, '/igvc/maneuver_command', self._on_maneuver_command, 10)
        self.create_subscription(
            String, '/igvc/yolo_prediction', self._on_yolo_prediction, 10)
        self.create_subscription(
            DetectionArray, detection_topic, self._on_detections, 10)
        self.create_subscription(
            Image, camera_topic, self._on_image, 10)

        # ---- service clients ---------------------------------------------
        self._clients: dict[str, Any] = {}
        for svc_name in (
            'arm', 'start', 'pause', 'resume',
            'abort', 'safe_stop', 'complete_maneuver', 'reconfigure',
        ):
            self._clients[svc_name] = self.create_client(
                Trigger, f'/{task_runner_node}/{svc_name}')

        self._set_params_client = self.create_client(
            SetParameters,
            f'/{task_runner_node}/set_parameters',
        )

    # ------------------------------------------------------------------
    # Subscription callbacks
    # ------------------------------------------------------------------

    def _on_task_status(self, msg: String) -> None:
        try:
            data = json.loads(msg.data)
            self.status_updated.emit(data)
        except (json.JSONDecodeError, TypeError):
            pass

    def _on_maneuver_command(self, msg: String) -> None:
        try:
            data = json.loads(msg.data)
            self.maneuver_updated.emit(data)
        except (json.JSONDecodeError, TypeError):
            pass

    def _on_yolo_prediction(self, msg: String) -> None:
        try:
            data = json.loads(msg.data)
            self.prediction_updated.emit(data)
        except (json.JSONDecodeError, TypeError):
            pass

    def _on_detections(self, msg: DetectionArray) -> None:
        self._latest_detections = list(msg.detections)

    def _on_image(self, msg: Image) -> None:
        try:
            bgr = self._bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            if self._latest_detections:
                bgr = draw_detections(bgr, self._latest_detections)
            self.image_updated.emit(bgr)
        except Exception:  # noqa: BLE001
            pass

    # ------------------------------------------------------------------
    # Service call helpers (fire-and-forget, results emitted as signals)
    # ------------------------------------------------------------------

    def _call_trigger(self, name: str) -> None:
        client = self._clients.get(name)
        if client is None or not client.service_is_ready():
            self.service_feedback.emit(
                f'{name}: service not available')
            return
        future = client.call_async(Trigger.Request())
        future.add_done_callback(
            lambda f: self._trigger_done(name, f))

    def _trigger_done(self, name: str, future) -> None:
        try:
            result = future.result()
            status = 'OK' if result.success else 'FAIL'
            self.service_feedback.emit(f'{name}: {status} — {result.message}')
        except Exception as exc:  # noqa: BLE001
            self.service_feedback.emit(f'{name}: exception — {exc}')

    # Public control methods called from the GUI thread.
    def call_arm(self) -> None:
        self._call_trigger('arm')

    def call_start(self) -> None:
        self._call_trigger('start')

    def call_pause(self) -> None:
        self._call_trigger('pause')

    def call_resume(self) -> None:
        self._call_trigger('resume')

    def call_abort(self) -> None:
        self._call_trigger('abort')

    def call_safe_stop(self) -> None:
        self._call_trigger('safe_stop')

    def call_complete_maneuver(self) -> None:
        self._call_trigger('complete_maneuver')

    def select_task(self, task_id: str) -> None:
        """Set the selected_task parameter on the runner then reconfigure."""
        if not self._set_params_client.service_is_ready():
            self.service_feedback.emit(
                'select_task: set_parameters service not available')
            return

        param_value = ParameterValue(
            type=ParameterType.PARAMETER_STRING,
            string_value=task_id,
        )
        param = RclParameter(name='selected_task', value=param_value)
        req = SetParameters.Request(parameters=[param])
        future = self._set_params_client.call_async(req)
        future.add_done_callback(
            lambda f: self._set_param_done(task_id, f))

    def _set_param_done(self, task_id: str, future) -> None:
        try:
            result = future.result()
            if result.results and result.results[0].successful:
                self.service_feedback.emit(
                    f'selected_task set to {task_id!r} — calling reconfigure')
                self._call_trigger('reconfigure')
            else:
                reason = (result.results[0].reason
                          if result.results else 'unknown')
                self.service_feedback.emit(
                    f'set_parameters failed: {reason}')
        except Exception as exc:  # noqa: BLE001
            self.service_feedback.emit(f'set_parameters: exception — {exc}')


# ---------------------------------------------------------------------------
# Background Qt thread that spins the ROS2 node
# ---------------------------------------------------------------------------

class RosThread(QThread):
    """Spins ``node`` in a background daemon thread."""

    def __init__(self, node: TaskGuiNode) -> None:
        super().__init__()
        self.node = node
        self.setDaemon(True)

    def run(self) -> None:
        rclpy.spin(self.node)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(args: list[str] | None = None) -> None:
    """Launch the IGVC task GUI application."""
    # Import here to avoid pulling Qt into every process that imports this module.
    from PyQt5.QtWidgets import QApplication  # noqa: PLC0415
    from igvc_task_gui.main_window import MainWindow  # noqa: PLC0415

    rclpy.init(args=args)

    # Read optional overrides from ROS2 params before Qt takes over.
    # Use a temporary node just for parameter reading.
    _tmp = rclpy.create_node('_igvc_task_gui_param_reader')
    _tmp.declare_parameter('task_runner_node', 'igvc_task_runner')
    _tmp.declare_parameter('camera_topic', '/front_zed_camera_x/rgb/image_raw')
    _tmp.declare_parameter('detection_topic', '/detections')
    runner_node_name = _tmp.get_parameter('task_runner_node').value
    camera_topic = _tmp.get_parameter('camera_topic').value
    detection_topic = _tmp.get_parameter('detection_topic').value
    _tmp.destroy_node()

    node = TaskGuiNode(
        task_runner_node=runner_node_name,
        camera_topic=camera_topic,
        detection_topic=detection_topic,
    )

    ros_thread = RosThread(node)
    ros_thread.start()

    # Find igvc_task_runner's share directory so the GUI can load task profiles.
    try:
        task_config_dir = (
            get_package_share_directory('igvc_task_runner')
            + '/config/tasks'
        )
    except Exception:  # noqa: BLE001
        task_config_dir = ''

    app = QApplication(sys.argv)
    app.setApplicationName('IGVC Task Control')
    window = MainWindow(node=node, task_config_dir=task_config_dir)
    window.show()

    try:
        sys.exit(app.exec_())
    finally:
        node.destroy_node()
        rclpy.shutdown()
