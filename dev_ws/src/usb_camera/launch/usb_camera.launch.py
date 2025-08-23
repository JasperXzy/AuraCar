#!/usr/bin/env python3

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration

def generate_launch_description():
    # 声明启动参数
    device_path_arg = DeclareLaunchArgument(
        'device_path',
        default_value='/dev/video0',
        description='USB摄像头设备路径'
    )
    
    width_arg = DeclareLaunchArgument(
        'width',
        default_value='1920',
        description='图像宽度'
    )
    
    height_arg = DeclareLaunchArgument(
        'height',
        default_value='1080',
        description='图像高度'
    )
    
    fps_arg = DeclareLaunchArgument(
        'fps',
        default_value='30',
        description='帧率'
    )
    
    pixel_format_arg = DeclareLaunchArgument(
        'pixel_format',
        default_value='mjpeg',
        description='像素格式 (mjpeg 或 yuyv)'
    )
    
    # 创建USB摄像头节点
    usb_camera_node = Node(
        package='usb_camera',
        executable='usb_camera_node',
        name='usb_camera_node',
        output='screen',
        parameters=[{
            'device_path': LaunchConfiguration('device_path'),
            'width': LaunchConfiguration('width'),
            'height': LaunchConfiguration('height'),
            'fps': LaunchConfiguration('fps'),
            'pixel_format': LaunchConfiguration('pixel_format'),
        }],
        remappings=[
            ('camera/image_raw', 'camera/image_raw'),
        ]
    )
    
    return LaunchDescription([
        device_path_arg,
        width_arg,
        height_arg,
        fps_arg,
        pixel_format_arg,
        usb_camera_node,
    ])
