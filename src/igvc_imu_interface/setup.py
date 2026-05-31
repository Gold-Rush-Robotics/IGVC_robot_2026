from setuptools import find_packages, setup

package_name = 'igvc_imu_interface'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='nitin-5090',
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
            'l3gd20_heading_node = igvc_imu_interface.l3gd20_heading_node:main',
            'imu_interface_node = igvc_imu_interface.l3gd20_heading_node:main',
        ],
    },
)
