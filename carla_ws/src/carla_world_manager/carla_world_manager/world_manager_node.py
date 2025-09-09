import os
from typing import Optional
import rclpy
from rclpy.node import Node
from rclpy.timer import Timer
from std_srvs.srv import SetBool, Trigger
from std_msgs.msg import UInt64

try:
    import carla
except ImportError as exc:
    raise SystemExit(
        "carla not found. Please install with 'pip install carla' or set PYTHONPATH to CARLA egg."
    ) from exc


# 默认物理步长（秒），约 30Hz
DEFAULT_FIXED_DELTA_SECONDS = 0.0333333333


def get_env_or_default(name: str, default_value: str) -> str:
    """读取环境变量，若不存在则返回默认值

    - name: 环境变量名称
    - default_value: 变量不存在时使用的默认字符串
    """
    value = os.environ.get(name)
    return value if value is not None else default_value


class CarlaWorldManager(Node):
    """CARLA 世界管理节点

    - 连接至 CARLA 服务器并按需加载地图
    - 管理同步/异步模式与固定时间步长
    - 发布世界 tick 计数话题
    - 暴露服务切换同步模式与重置世界
    """

    def __init__(self) -> None:
        super().__init__('carla_world_manager')

        # 参数
        self.host = self.declare_parameter(
            'host', get_env_or_default('CARLA_HOST', '127.0.0.1')
        ).get_parameter_value().string_value
        self.port = self.declare_parameter(
            'port', int(get_env_or_default('CARLA_PORT', '2000'))
        ).get_parameter_value().integer_value
        self.sync_mode = self.declare_parameter('sync_mode', False).get_parameter_value().bool_value
        self.fixed_delta_seconds = self.declare_parameter(
            'fixed_delta_seconds', DEFAULT_FIXED_DELTA_SECONDS
        ).get_parameter_value().double_value
        self.town = self.declare_parameter('town', '').get_parameter_value().string_value

        # 连接 CARLA 服务器
        self.client = carla.Client(self.host, self.port)
        self.client.set_timeout(5.0)
        self.world = self.client.get_world()

        # 若指定了城市场景且与当前不一致，则加载目标地图
        if self.town and self.world.get_map().name != self.town:
            self.get_logger().info(f'Loading town {self.town} ...')
            self.world = self.client.load_world(self.town)

        # 发布世界 tick 计数（UInt64）
        self.tick_pub = self.create_publisher(UInt64, '/carla/world/tick', 10)
        self.tick_counter: int = 0

        # 在同步模式下使用定时器推进世界 tick
        self.tick_timer: Optional[Timer] = None
        self.apply_settings(self.sync_mode, self.fixed_delta_seconds)

        # 服务：切换同步模式与重置世界
        self.srv_set_sync = self.create_service(SetBool, 'set_sync_mode', self.handle_set_sync_mode)
        self.srv_reset = self.create_service(Trigger, 'reset_world', self.handle_reset_world)

        self.get_logger().info(
            f'CarlaWorldManager connected to {self.host}:{self.port}, '
            f'sync_mode={self.sync_mode}, fixed_dt={self.fixed_delta_seconds}'
        )

    def apply_settings(self, sync_mode: bool, fixed_dt: float) -> None:
        """应用 CARLA 世界设置，并根据同步模式重建定时器

        - sync_mode: 是否使用同步模式
        - fixed_dt: 同步模式下的固定时间步长（秒）
        """
        settings = self.world.get_settings()
        settings.synchronous_mode = bool(sync_mode)
        if sync_mode:
            settings.fixed_delta_seconds = float(fixed_dt)
        else:
            # 异步模式下由仿真引擎自行推进
            settings.fixed_delta_seconds = None
        self.world.apply_settings(settings)

        # 在切换模式时重建定时器
        if self.tick_timer is not None:
            self.tick_timer.cancel()
            self.tick_timer = None
        if sync_mode:
            period = float(fixed_dt) if fixed_dt and fixed_dt > 0.0 else DEFAULT_FIXED_DELTA_SECONDS
            self.tick_timer = self.create_timer(period, self._on_tick_timer)

    def _on_tick_timer(self) -> None:
        """同步模式下由定时器回调触发世界推进一次"""
        try:
            self.world.tick()
            self.tick_counter += 1
            msg = UInt64()
            msg.data = self.tick_counter
            self.tick_pub.publish(msg)
        except Exception as exc:
            self.get_logger().warning(f'world.tick() failed: {exc}')

    def handle_set_sync_mode(self, request: SetBool.Request, response: SetBool.Response) -> SetBool.Response:
        """服务：切换同步模式

        请求体 request.data 为布尔值；成功后会重建定时器并应用设置
        """
        try:
            self.sync_mode = bool(request.data)
            self.apply_settings(self.sync_mode, self.fixed_delta_seconds)
            response.success = True
            response.message = f'sync_mode set to {self.sync_mode}'
        except Exception as exc:  # noqa: BLE001
            response.success = False
            response.message = f'failed: {exc}'
        return response

    def handle_reset_world(self, _request: Trigger.Request, response: Trigger.Response) -> Trigger.Response:
        """服务：重置并重载世界

        会尽量保持当前地图（或强制加载参数指定的地图），并重新应用设置
        """
        try:
            # 保持相同地图
            current_map = self.world.get_map().name
            self.world = self.client.reload_world(False)
            if self.town and current_map != self.town:
                self.world = self.client.load_world(self.town)
            # 重新应用设置与计数
            self.apply_settings(self.sync_mode, self.fixed_delta_seconds)
            self.tick_counter = 0
            response.success = True
            response.message = 'world reloaded'
        except Exception as exc:
            response.success = False
            response.message = f'failed: {exc}'
        return response


def main() -> None:
    """初始化 rclpy 并运行节点"""
    rclpy.init()
    node = CarlaWorldManager()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
