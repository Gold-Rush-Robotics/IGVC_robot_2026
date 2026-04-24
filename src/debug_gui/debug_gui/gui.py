import sys
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QHBoxLayout
from PyQt5.QtCore import Qt, QObject, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap
from sensor_msgs.msg import Image
from functools import partial

import threading

class ImageSignal(QObject):
    signal = pyqtSignal(object, int)

class JointStatePublisher(Node):
    def __init__(self):
        super().__init__('joint_state_publisher')
        self.publisher = self.create_publisher(JointState, 'isaac_joint_commands', 10)
        # Subscribe to isaac_joint_commands to update joint positions in the GUI
        self.camera_subscribers = []
        self.create_subscribers()
        self.app = None
        

    def add_app(self, app):
        self.app = app
        self.image_signal = ImageSignal()
        self.image_signal.signal.connect(self.app.update_image)

    def create_subscribers(self):
        # topics = [ "left_zed_camera_x/zed_node/rgb/color/rect/image","/front_zed_camera_x/zed_node/rgb/color/rect/image","right_zed_camera_x/zed_node/rgb/color/rect/image"]
        # topics = [ "/lane_debug/cam0/overlay", "/lane_debug/cam1/overlay", "/lane_debug/cam2/overlay" ]
        topics = [ "/lane_debug/cam0/overlay" ]
        # topics = [ "/debug/stitched_lanes" ]
        for i, topic in enumerate(topics):
            callback = partial(self.image_callback, index=i)
            subscriber = self.create_subscription(Image, topic, callback, 10)
            self.camera_subscribers.append(subscriber)

    def image_callback(self, msg, index):
        self.image_signal.signal.emit(msg, index)


class JointControlGUI(QWidget):
    def __init__(self, node):
        super().__init__()
        self.node = node
        self.initUI()
        self.ros2_thread = threading.Thread(target=self.spin_ros2_node)
        self.ros2_thread.start()

    def spin_ros2_node(self):
        rclpy.spin(self.node)

    def closeEvent(self, event):
        # Override the close event to ensure proper shutdown
        rclpy.shutdown()  # Shutdown ROS2
        self.ros2_thread.join()  # Wait for the ROS2 thread to finish
        event.accept()  # Proceed with the shutdown

    def initUI(self):
        # Main layout
        mainLayout = QHBoxLayout()

        self.image_labels = []
        
        for i in range(len(self.node.camera_subscribers)):
            image_label = QLabel()
            # image_label.setFixedSize(640, 480)
            # self.layout.addWidget(image_label)
            mainLayout.addWidget(image_label)
            self.image_labels.append(image_label)

        self.setLayout(mainLayout)

        self.setWindowTitle('MyCobot Joint Control GUI')
        self.show()

    def update_image(self, msg, label_index):
        qt_image = self.convert_ros_image_to_qpixmap(msg)
        if qt_image is None:
            return
        self.image_labels[label_index].setPixmap(qt_image)
        
    def convert_ros_image_to_qpixmap(self, msg):
        if msg.encoding == 'rgb8':
            image = QImage(msg.data, msg.width, msg.height, msg.step, QImage.Format_RGB888)
        elif msg.encoding == 'bgr8':
            image = QImage(msg.data, msg.width, msg.height, msg.step, QImage.Format_RGB888).rgbSwapped()
        elif msg.encoding == 'rgba8':
            image = QImage(msg.data, msg.width, msg.height, msg.step, QImage.Format_RGBA8888)
        elif msg.encoding == 'bgra8':
            image = QImage(msg.data, msg.width, msg.height, msg.step, QImage.Format_ARGB32)
        elif msg.encoding in ('mono8', '8UC1'):
            image = QImage(msg.data, msg.width, msg.height, msg.step, QImage.Format_Grayscale8)
        else:
            self.node.get_logger().warn(f'Unsupported image encoding: {msg.encoding}')
            return None

        return QPixmap.fromImage(image.copy().scaled(640, 480, Qt.KeepAspectRatio))

def main(args=None):
    rclpy.init(args=args)
    app = QApplication([])
    node = JointStatePublisher()
    ex = JointControlGUI(node)
    node.add_app(ex)
    sys.exit(app.exec_())
    # rclpy.shutdown()

if __name__ == '__main__':
    main()