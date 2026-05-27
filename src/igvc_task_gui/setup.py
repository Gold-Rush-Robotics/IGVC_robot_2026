from glob import glob

from setuptools import find_packages, setup

package_name = 'igvc_task_gui'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools', 'PyYAML'],
    zip_safe=True,
    maintainer='ubuntu',
    maintainer_email='nchan18@outlook.com',
    description='PyQt5 operator GUI for IGVC task selection and state machine monitoring.',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'task_gui = igvc_task_gui.task_gui_node:main',
        ],
    },
)
