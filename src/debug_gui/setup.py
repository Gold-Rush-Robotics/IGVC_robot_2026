from setuptools import setup

package_name = 'debug_gui'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='nchan18',
    maintainer_email='nchan18@outlook.com',
    description='GUI Robot Control Debugger',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'debugger = debug_gui.gui:main'
        ],
    },
)
