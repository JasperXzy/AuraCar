#!/usr/bin/env python3

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    # 获取包的路径
    pkg_share = FindPackageShare('image_compressed_to_nv12')
    
    # 配置文件路径
    config_file = PathJoinSubstitution([
        pkg_share,
        '..',
        '..',
        '..',
        '..',
        'config',
        'image_compressed_to_nv12.yaml'
    ])
    
    # 创建图像压缩转NV12节点
    image_compressed_to_nv12_node = Node(
        package='image_compressed_to_nv12',
        executable='image_compressed_to_nv12_node',
        name='image_compressed_to_nv12_node',
        output='screen',
        parameters=[config_file],
        remappings=[
            ('image_compressed', 'image_compressed'),
            ('image_raw', 'image_raw'),
        ]
    )
    
    return LaunchDescription([
        image_compressed_to_nv12_node,
    ])
