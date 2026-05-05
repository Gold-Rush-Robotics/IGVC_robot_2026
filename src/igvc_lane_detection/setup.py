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
    # NOTE: on the Jetson AGX Orin target, ``torch`` must be installed
    # manually from the NVIDIA JetPack wheel index
    # (https://developer.download.nvidia.com/compute/redist/jp/) — the
    # stock PyPI wheel has no CUDA on aarch64.  It is listed here for
    # completeness; pip will be a no-op if the Jetson wheel is already
    # installed into the active Python environment.
    install_requires=[
        'setuptools',
        'numpy<2',
        'ultralytics',
        'opencv-python-headless',
        'torch',
    ],
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
            'lane_segmentation_node = igvc_lane_detection.lane_segmentation:main',
        ],
    },
)
