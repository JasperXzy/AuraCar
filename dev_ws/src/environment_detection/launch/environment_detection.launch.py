#!/usr/bin/env python3

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    params_file_arg = DeclareLaunchArgument(
        'params_file', default_value='config/environment_detection.yaml',
        description='YAML 参数文件路径（ros__parameters）')

    node = Node(
        package='environment_detection',
        executable='environment_detection_node',
        name='environment_detection',
        output='screen',
        emulate_tty=True,
        arguments=['--ros-args', '--params-file', LaunchConfiguration('params_file')]
    )

    return LaunchDescription([
        params_file_arg,
        node,
    ])
