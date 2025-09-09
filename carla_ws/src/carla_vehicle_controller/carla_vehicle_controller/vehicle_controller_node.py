import math
import os
import time
from typing import Optional

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.time import Time
from std_srvs.srv import SetBool
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from tf2_ros import TransformBroadcaster
from geometry_msgs.msg import TransformStamped

try:
    from ackermann_msgs.msg import AckermannDriveStamped  # type: ignore
except Exception:
    AckermannDriveStamped = None  # type: ignore

try:
    import carla  # type: ignore
except ImportError as exc:
    raise SystemExit(
        "carla package not found. Please install with 'pip install carla' or set PYTHONPATH to CARLA egg."
    ) from exc


def get_env_or_default(name: str, default_value: str) -> str:
    value = os.environ.get(name)
    return value if value is not None else default_value


def euler_to_quaternion(roll: float, pitch: float, yaw: float):
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    return x, y, z, w


class VehicleControllerNode(Node):
    """车辆控制节点

    - 连接 CARLA 并查找/生成车辆实体
    - 订阅速度/转向命令（Twist 或 Ackermann）并应用车辆控制
    - 发布里程计与 TF 信息
    - 提供启用/关闭自动驾驶服务
    """
    def __init__(self) -> None:
        super().__init__('vehicle_controller')

        # 参数
        self.host = self.declare_parameter('host', get_env_or_default('CARLA_HOST', '127.0.0.1')).get_parameter_value().string_value
        self.port = self.declare_parameter('port', int(get_env_or_default('CARLA_PORT', '2000'))).get_parameter_value().integer_value
        self.vehicle_id = self.declare_parameter('vehicle_id', 'ego').get_parameter_value().string_value
        self.vehicle_blueprint = self.declare_parameter('vehicle_blueprint', 'vehicle.tesla.model3').get_parameter_value().string_value
        self.spawn_point_index = self.declare_parameter('spawn_point_index', 0).get_parameter_value().integer_value
        self.autopilot = self.declare_parameter('autopilot', False).get_parameter_value().bool_value
        self.control_mode = self.declare_parameter('control_mode', 'manual').get_parameter_value().string_value
        self.max_steer = float(self.declare_parameter('max_steer', 0.8).get_parameter_value().double_value)
        self.max_throttle = float(self.declare_parameter('max_throttle', 0.6).get_parameter_value().double_value)
        self.max_brake = float(self.declare_parameter('max_brake', 0.8).get_parameter_value().double_value)
        self.max_speed = float(self.declare_parameter('max_speed', 20.0).get_parameter_value().double_value)
        self.cmd_timeout_ms = int(self.declare_parameter('cmd_timeout_ms', 200).get_parameter_value().integer_value)

        # 连接 CARLA
        self.client = carla.Client(self.host, self.port)
        self.client.set_timeout(5.0)
        self.world = self.client.get_world()

        # 查找或生成车辆
        self.vehicle: carla.Vehicle = self._find_or_spawn_vehicle()

        # 自动驾驶开关
        self.vehicle.set_autopilot(bool(self.autopilot or self.control_mode == 'autopilot'))

        # 接口：订阅命令，服务，发布里程计与 TF
        if AckermannDriveStamped is not None:
            self.sub_ack = self.create_subscription(AckermannDriveStamped, 'cmd_ackermann', self._on_cmd_ackermann, 10)
        self.sub_twist = self.create_subscription(Twist, 'cmd_vel', self._on_cmd_vel, 10)
        self.srv_autopilot = self.create_service(SetBool, 'set_autopilot', self._handle_set_autopilot)
        self.pub_odom = self.create_publisher(Odometry, 'odom', 10)
        self.tf_broadcaster = TransformBroadcaster(self)

        self.last_cmd_time: Time = self.get_clock().now()
        self.create_timer(0.05, self._on_50ms)

        self.get_logger().info(
            f'VehicleController started for vehicle_id={self.vehicle_id}, blueprint={self.vehicle_blueprint}, spawn_index={self.spawn_point_index}'
        )

    def _find_or_spawn_vehicle(self) -> 'carla.Vehicle':
        """查找 role_name 匹配的车辆 不存在则按参数生成"""
        # 先按 role_name 查找
        actors = self.world.get_actors().filter('vehicle.*')
        for actor in actors:
            try:
                if actor.attributes.get('role_name') == self.vehicle_id:
                    return actor  # type: ignore[return-value]
            except Exception:
                continue
        # 否则生成新车辆
        bp_lib = self.world.get_blueprint_library()
        try:
            bp = bp_lib.find(self.vehicle_blueprint)
        except Exception:
            # 回退：优先 model3 其后任意车辆
            candidates = bp_lib.filter('vehicle.*model3*')
            if not candidates:
                candidates = bp_lib.filter('vehicle.*')
            if not candidates:
                raise RuntimeError('No vehicle blueprints found')
            bp = candidates[0]
        bp.set_attribute('role_name', self.vehicle_id)

        spawn_points = self.world.get_map().get_spawn_points()
        index = int(self.spawn_point_index) if spawn_points else 0
        transform = spawn_points[index % len(spawn_points)] if spawn_points else carla.Transform()

        vehicle = self.world.try_spawn_actor(bp, transform)
        if vehicle is None:
            # 回退：再次使用同一位姿强制生成
            vehicle = self.world.spawn_actor(bp, transform)
        return vehicle  # type: ignore[return-value]

    def _apply_control(self, throttle: float, steer: float, brake: float = 0.0) -> None:
        """裁剪并应用车辆控制指令"""
        throttle = max(0.0, min(self.max_throttle, float(throttle)))
        steer = max(-self.max_steer, min(self.max_steer, float(steer)))
        brake = max(0.0, min(self.max_brake, float(brake)))
        control = carla.VehicleControl(
            throttle=throttle,
            steer=steer,
            brake=brake,
            hand_brake=False,
            reverse=False,
            manual_gear_shift=False,
        )
        try:
            self.vehicle.apply_control(control)
        except Exception as exc:  # noqa: BLE001
            self.get_logger().warning(f'apply_control failed: {exc}')

    def _on_cmd_ackermann(self, msg) -> None:
        """接收 Ackermann 命令并近似映射到油门/转向"""
        self.last_cmd_time = self.get_clock().now()
        # Map ackermann speed to throttle (naive)
        target_speed = float(getattr(msg.drive, 'speed', 0.0))
        steering = float(getattr(msg.drive, 'steering_angle', 0.0))
        throttle = max(0.0, min(1.0, target_speed / max(0.1, self.max_speed)))
        self._apply_control(throttle=throttle, steer=steering / max(0.001, self.max_steer))

    def _on_cmd_vel(self, msg: Twist) -> None:
        """接收 Twist 命令并近似映射到油门/转向"""
        self.last_cmd_time = self.get_clock().now()
        target_speed = float(msg.linear.x)
        steering_z = float(msg.angular.z)
        throttle = max(0.0, min(1.0, target_speed / max(0.1, self.max_speed)))
        self._apply_control(throttle=throttle, steer=steering_z)

    def _handle_set_autopilot(self, request: SetBool.Request, response: SetBool.Response) -> SetBool.Response:
        """服务：启用/关闭车辆自动驾驶"""
        try:
            self.vehicle.set_autopilot(bool(request.data))
            response.success = True
            response.message = f'autopilot set to {request.data}'
        except Exception as exc:  # noqa: BLE001
            response.success = False
            response.message = f'failed: {exc}'
        return response

    def _on_50ms(self) -> None:
        # 失效保护：命令超时则置 0 控制
        now = self.get_clock().now()
        elapsed_ns = (now - self.last_cmd_time).nanoseconds
        if elapsed_ns > int(self.cmd_timeout_ms) * 1_000_000:
            self._apply_control(throttle=0.0, steer=0.0, brake=0.0)

        # 以 20Hz 发布里程计与 TF
        try:
            transform = self.vehicle.get_transform()
            velocity = self.vehicle.get_velocity()
        except Exception as exc:  # noqa: BLE001
            self.get_logger().warning(f'get_transform failed: {exc}')
            return

        # 坐标转换：CARLA (X前 Y右 Z上 左手) -> ROS (X前 Y左 Z上)
        x = float(transform.location.x)
        y = float(-transform.location.y)
        z = float(transform.location.z)
        roll = math.radians(-transform.rotation.roll)
        pitch = math.radians(transform.rotation.pitch)
        yaw = math.radians(-transform.rotation.yaw)
        qx, qy, qz, qw = euler_to_quaternion(roll, pitch, yaw)

        odom = Odometry()
        odom.header.stamp = now.to_msg()
        odom.header.frame_id = 'map'
        odom.child_frame_id = 'base_link'
        odom.pose.pose.position.x = x
        odom.pose.pose.position.y = y
        odom.pose.pose.position.z = z
        odom.pose.pose.orientation.x = qx
        odom.pose.pose.orientation.y = qy
        odom.pose.pose.orientation.z = qz
        odom.pose.pose.orientation.w = qw
        odom.twist.twist.linear.x = float(velocity.x)
        odom.twist.twist.linear.y = float(-velocity.y)
        odom.twist.twist.linear.z = float(velocity.z)
        self.pub_odom.publish(odom)

        t = TransformStamped()
        t.header.stamp = now.to_msg()
        t.header.frame_id = 'map'
        t.child_frame_id = 'base_link'
        t.transform.translation.x = x
        t.transform.translation.y = y
        t.transform.translation.z = z
        t.transform.rotation.x = qx
        t.transform.rotation.y = qy
        t.transform.rotation.z = qz
        t.transform.rotation.w = qw
        self.tf_broadcaster.sendTransform(t)


def main() -> None:
    """初始化 rclpy 并运行节点"""
    rclpy.init()
    node = VehicleControllerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
