from setuptools import find_packages, setup

package_name = 'igvc_lane_detection'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools', 'numpy<2', 'ultralytics', 'opencv-python-headless'],
    zip_safe=True,
    maintainer='ubuntu',
    maintainer_email='nchan18@outlook.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'lane_detection_node = igvc_lane_detection.lane_detection:main',
            'navigation_node = igvc_lane_detection.navigator:main',
            'localization_node = igvc_lane_detection.localization:main',
            'odom_tf_bridge_node = igvc_lane_detection.odom_tf_bridge:main',
            'multi_camera_lane_detection_node = igvc_lane_detection.multi_camera_lane_detection:main',
        ],
    },
)
