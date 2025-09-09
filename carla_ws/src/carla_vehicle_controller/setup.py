from setuptools import find_packages, setup

package_name = 'carla_vehicle_controller'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='JasperXzy',
    maintainer_email='jasper.zhengyi.xu@gmail.com',
    description='A simple vehicle controller package for Carla',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'vehicle_controller_node = carla_vehicle_controller.vehicle_controller_node:main',
        ],
    },
)
