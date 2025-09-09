import os
from typing import Optional
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage, CameraInfo

try:
    import carla  # type: ignore
except ImportError as exc:
    raise SystemExit(
        "carla package not found. Please install with 'pip install carla' or set PYTHONPATH to CARLA egg."
    ) from exc

try:
    import cv2  # type: ignore
except ImportError as exc:
    raise SystemExit(
        "opencv-python not found. Please install with 'sudo apt install python3-opencv' or 'pip install opencv-python'."
    ) from exc


def get_env_or_default(name: str, default_value: str) -> str:
    value = os.environ.get(name)
    return value if value is not None else default_value


class FrontCameraNode(Node):
    """前视相机节点

    - 连接 CARLA 并附着 RGB 相机至指定车辆
    - 发布压缩图像 `image_compressed` 与相机内参 `camera_info`
    """
    def __init__(self) -> None:
        super().__init__('front_camera')

        # 参数
        self.host = self.declare_parameter('host', get_env_or_default('CARLA_HOST', '127.0.0.1')).get_parameter_value().string_value
        self.port = self.declare_parameter('port', int(get_env_or_default('CARLA_PORT', '2000'))).get_parameter_value().integer_value
        self.vehicle_id = self.declare_parameter('vehicle_id', 'ego').get_parameter_value().string_value
        self.width = self.declare_parameter('width', int(get_env_or_default('CAM_WIDTH', '1280'))).get_parameter_value().integer_value
        self.height = self.declare_parameter('height', int(get_env_or_default('CAM_HEIGHT', '720'))).get_parameter_value().integer_value
        self.fov = float(self.declare_parameter('fov', 90.0).get_parameter_value().double_value)
        self.fps = int(self.declare_parameter('fps', 30).get_parameter_value().integer_value)
        self.quality = int(self.declare_parameter('quality', 90).get_parameter_value().integer_value)
        self.frame_id = self.declare_parameter('frame_id', 'camera_front').get_parameter_value().string_value

        # 发布器：压缩图像与相机内参
        self.pub_img = self.create_publisher(CompressedImage, 'image_compressed', 10)
        self.pub_info = self.create_publisher(CameraInfo, 'camera_info', 10)

        # 连接 CARLA 服务器
        self.client = carla.Client(self.host, self.port)
        self.client.set_timeout(5.0)
        self.world = self.client.get_world()

        # 根据 role_name 查找目标车辆
        self.vehicle = self._wait_and_find_vehicle(self.vehicle_id, timeout_sec=5.0)

        # 生成并附着相机到车辆前部
        bp_lib = self.world.get_blueprint_library()
        cam_bp = bp_lib.find('sensor.camera.rgb')
        cam_bp.set_attribute('image_size_x', str(self.width))
        cam_bp.set_attribute('image_size_y', str(self.height))
        cam_bp.set_attribute('fov', str(self.fov))
        if self.fps > 0:
            cam_bp.set_attribute('sensor_tick', str(1.0 / float(self.fps)))
        cam_tf = carla.Transform(carla.Location(x=1.5, z=1.6))
        self.sensor = self.world.spawn_actor(cam_bp, cam_tf, attach_to=self.vehicle)
        self.sensor.listen(self._on_image)

        self.get_logger().info(
            f'FrontCamera attached to vehicle_id={self.vehicle_id}, {self.width}x{self.height}@{self.fps} fov={self.fov}, jpeg={self.quality}'
        )

    def _wait_and_find_vehicle(self, role_name: str, timeout_sec: float) -> 'carla.Vehicle':
        """按 role_name 等待并查找车辆 若超时则抛出异常"""
        import time
        deadline = time.time() + float(timeout_sec)
        while time.time() < deadline:
            actors = self.world.get_actors().filter('vehicle.*')
            for actor in actors:
                try:
                    if actor.attributes.get('role_name') == role_name:
                        return actor  # type: ignore[return-value]
                except Exception:
                    continue
            time.sleep(0.2)
        raise RuntimeError(f'No vehicle found with role_name={role_name} after {timeout_sec}s.')

    def _on_image(self, image: 'carla.Image') -> None:
        """相机回调 将 CARLA 原始 BGRA 图像编码为 JPEG 并发布"""
        np_img = np.frombuffer(image.raw_data, dtype=np.uint8).reshape((image.height, image.width, 4))
        bgr = np_img[:, :, :3]
        ok, enc = cv2.imencode('.jpg', bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(self.quality)])
        if not ok:
            return
        now = self.get_clock().now().to_msg()

        msg = CompressedImage()
        msg.header.stamp = now
        msg.header.frame_id = self.frame_id
        msg.format = 'jpeg'
        msg.data = enc.tobytes()
        self.pub_img.publish(msg)

        # 基于针孔相机模型计算内参矩阵的焦距 fx
        fx = self.width / (2.0 * np.tan(np.deg2rad(self.fov) / 2.0))
        info = CameraInfo()
        info.header = msg.header
        info.width = self.width
        info.height = self.height
        # K: 3x3 内参矩阵 (行优先一维展开)
        info.k = [fx, 0.0, self.width / 2.0, 0.0, fx, self.height / 2.0, 0.0, 0.0, 1.0]
        # P: 3x4 投影矩阵（此处无畸变与基线 采用简单针孔模型）
        info.p = [fx, 0.0, self.width / 2.0, 0.0, 0.0, fx, self.height / 2.0, 0.0, 0.0, 0.0, 1.0, 0.0]
        self.pub_info.publish(info)

    def destroy_node(self) -> None:
        """节点销毁时安全停止并释放相机传感器"""
        try:
            if getattr(self, 'sensor', None) is not None:
                self.sensor.stop()
                self.sensor.destroy()
        finally:
            super().destroy_node()


def main() -> None:
    """初始化 rclpy 并运行节点"""
    rclpy.init()
    node = FrontCameraNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
