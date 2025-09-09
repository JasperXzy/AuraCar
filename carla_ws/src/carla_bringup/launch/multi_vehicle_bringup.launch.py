from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
import yaml


def launch_setup(context, *args, **kwargs):
    """根据配置生成节点 Actions 列表

    - config: YAML 配置路径
    """
    config_path = LaunchConfiguration('config').perform(context)
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)

    actions = []

    # 世界管理节点参数
    world_params = cfg.get('world', {})
    actions.append(
        Node(
            package='carla_world_manager',
            executable='world_manager_node',
            name='carla_world_manager',
            output='screen',
            parameters=[world_params],
        )
    )

    # 车辆与其前视相机
    vehicles = cfg.get('vehicles', [])
    for vehicle in vehicles:
        vid = str(vehicle.get('id', 'veh'))
        ns = f'vehicle_{vid}'
        # 车辆控制参数
        vc_params = {
            'host': world_params.get('host', '127.0.0.1'),
            'port': world_params.get('port', 2000),
            'vehicle_id': vid,
            'vehicle_blueprint': vehicle.get('blueprint', 'vehicle.tesla.model3'),
            'spawn_point_index': int(vehicle.get('spawn_point_index', 0)),
            'autopilot': bool(vehicle.get('autopilot', False)),
        }
        actions.append(
            Node(
                package='carla_vehicle_controller',
                executable='vehicle_controller_node',
                name='vehicle_controller',
                namespace=ns,
                output='screen',
                parameters=[vc_params],
            )
        )

        # 前视相机参数
        cam = vehicle.get('front_camera', {})
        cam_params = {
            'host': world_params.get('host', '127.0.0.1'),
            'port': world_params.get('port', 2000),
            'vehicle_id': vid,
            'width': int(cam.get('width', 1280)),
            'height': int(cam.get('height', 720)),
            'fov': float(cam.get('fov', 90.0)),
            'fps': int(cam.get('fps', 30)),
            'quality': int(cam.get('quality', 90)),
            'frame_id': str(cam.get('frame_id', 'camera_front')),
        }
        actions.append(
            Node(
                package='carla_front_camera',
                executable='front_camera_node',
                name='front_camera',
                namespace=f'{ns}/front_camera',
                output='screen',
                parameters=[cam_params],
            )
        )

    return actions


def generate_launch_description():
    """生成 LaunchDescription 并声明配置参数路径"""
    # 默认配置文件路径
    default_config = '../../../../config/multi_vehicle.yaml'
    return LaunchDescription([
        DeclareLaunchArgument('config', default_value=default_config, description='Path to YAML config'),
        OpaqueFunction(function=launch_setup),
    ])
