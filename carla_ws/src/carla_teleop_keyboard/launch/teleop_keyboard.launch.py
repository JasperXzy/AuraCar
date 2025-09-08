from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    vehicle_id_arg = DeclareLaunchArgument('vehicle_id', default_value='ego')
    mode_arg = DeclareLaunchArgument('mode', default_value='twist')  # twist|ackermann

    return LaunchDescription([
        vehicle_id_arg,
        mode_arg,
        Node(
            package='carla_teleop_keyboard',
            executable='teleop_keyboard_node',
            name='teleop_keyboard',
            output='screen',
            parameters=[{
                'vehicle_id': LaunchConfiguration('vehicle_id'),
                'mode': LaunchConfiguration('mode'),
            }],
        ),
    ])
