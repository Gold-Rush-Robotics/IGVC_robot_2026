from glob import glob
from setuptools import find_packages, setup


package_name = 'igvc_task_runner'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.launch.py')),
        ('share/' + package_name + '/config', glob('config/*.yaml')),
        ('share/' + package_name + '/config/tasks', glob('config/tasks/*.yaml')),
    ],
    install_requires=['setuptools', 'PyYAML'],
    zip_safe=True,
    maintainer='ubuntu',
    maintainer_email='nchan18@outlook.com',
    description='Lifecycle-style task runner for IGVC autonomous runs.',
    license='TODO: License declaration',
    extras_require={
        'test': ['pytest'],
    },
    entry_points={
        'console_scripts': [
            'task_runner = igvc_task_runner.runner:main',
        ],
    },
)
