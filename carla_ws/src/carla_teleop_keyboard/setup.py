from setuptools import find_packages, setup

package_name = 'carla_teleop_keyboard'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/teleop_keyboard.launch.py']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='JasperXzy',
    maintainer_email='jasper.zhengyi.xu@gmail.com',
    description='A simple teleop package for Carla',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'teleop_keyboard_node = carla_teleop_keyboard.teleop_keyboard_node:main',
        ],
    },
)
