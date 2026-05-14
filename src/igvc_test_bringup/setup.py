from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'igvc_test_bringup'

# Gather all launch and config files
launch_files = glob(os.path.join('launch', '*.launch.py')) + glob(os.path.join('launch', '*.launch.yaml'))
config_files = glob(os.path.join('config', '*'))

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), launch_files),
        (os.path.join('share', package_name, 'config'), config_files),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ubuntu',
    maintainer_email='nchan18@outlook.com',
    description='IGVC test bringup and launch files',
    license='TODO: License declaration',
    entry_points={
        'console_scripts': [
            'sim_startup_helper = igvc_test_bringup.sim_startup_helper:main',
        ],
    },
)
