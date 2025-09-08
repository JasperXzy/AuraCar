from setuptools import find_packages, setup

# 打包配置（ament_python）：定义 ROS 2 Python 包 carla_world_manager
# 说明：
# - packages: 自动查找 carla_world_manager 下的 Python 包
# - data_files: 安装 package.xml 与资源索引，以便被 ROS 2 发现
# - entry_points: 注册可执行脚本 world_manager_node

package_name = 'carla_world_manager'

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
    description='A simple world manager package for Carla',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'world_manager_node = carla_world_manager.world_manager_node:main',
        ],
    },
)
