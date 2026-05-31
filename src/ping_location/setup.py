from setuptools import find_packages, setup

package_name = 'ping_location'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools', 'requests'],
    zip_safe=True,
    maintainer='IGVC Team',
    maintainer_email='pgovindmenon07@gmail.com',
    description='ROS 2 node for fetching GPS location from remote phone endpoints and computing midpoint',
    license='Apache-2.0',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'ping_location_node = ping_location.ping_location_node:main',
        ],
    },
)
