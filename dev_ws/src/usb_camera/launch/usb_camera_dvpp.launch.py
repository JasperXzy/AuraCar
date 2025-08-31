#!/usr/bin/env python3

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    # 获取包的路径
    pkg_share = FindPackageShare('usb_camera')
    
    # 配置文件路径
    config_file = PathJoinSubstitution([
        pkg_share,
        '..',
        '..',
        '..',
        '..',
        'config',
        'usb_camera_dvpp.yaml'
    ])
    
    # 创建DVPP优化的USB摄像头节点
    usb_camera_node_dvpp = Node(
        package='usb_camera',
        executable='usb_camera_node_dvpp',
        name='usb_camera_node_dvpp',
        output='screen',
        parameters=[config_file],
        remappings=[
            ('camera/image_raw', 'camera/image_raw'),
        ]
    )
    
    return LaunchDescription([
        usb_camera_node_dvpp,
    ])
