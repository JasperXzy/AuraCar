import math
import os
from typing import Optional

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo

try:
    import carla
except ImportError as exc:
    raise SystemExit(
        "carla package not found. Please install with 'pip install carla' or set PYTHONPATH to CARLA egg."
    ) from exc


def get_env_or_default(name: str, default_value: str) -> str:
    value = os.environ.get(name)
    return value if value is not None else default_value


class CarlaCameraNode(Node):
    def __init__(self) -> None:
        super().__init__('carla_camera_node')

        # Declare parameters with environment-variable fallbacks
        host = self.declare_parameter('host', get_env_or_default('CARLA_HOST', '127.0.0.1')).get_parameter_value().string_value
        port = self.declare_parameter('port', int(get_env_or_default('CARLA_PORT', '2000'))).get_parameter_value().integer_value
        width = self.declare_parameter('width', int(get_env_or_default('CAM_WIDTH', '1280'))).get_parameter_value().integer_value
        height = self.declare_parameter('height', int(get_env_or_default('CAM_HEIGHT', '720'))).get_parameter_value().integer_value
        fov = self.declare_parameter('fov', float(get_env_or_default('CAM_FOV', '90.0'))).get_parameter_value().double_value
        topic_ns = self.declare_parameter('topic_ns', get_env_or_default('TOPIC_NS', 'camera/front')).get_parameter_value().string_value

        image_topic = f'{topic_ns}/image_raw'
        info_topic = f'{topic_ns}/camera_info'

        self.publisher_image = self.create_publisher(Image, image_topic, 10)
        self.publisher_info = self.create_publisher(CameraInfo, info_topic, 10)

        self.width = width
        self.height = height
        self.fov = fov

        # Connect to CARLA
        client = carla.Client(host, port)
        client.set_timeout(5.0)
        self.world = client.get_world()
        blueprint_library = self.world.get_blueprint_library()

        # Spawn or find a vehicle to attach the camera
        vehicle_blueprints = blueprint_library.filter('vehicle.*model3*')
        vehicle_bp = vehicle_blueprints[0] if vehicle_blueprints else blueprint_library.filter('vehicle.*')[0]

        spawn_points = self.world.get_map().get_spawn_points()
        spawn_point = spawn_points[0] if spawn_points else carla.Transform()

        self.vehicle: Optional[carla.Actor] = self.world.try_spawn_actor(vehicle_bp, spawn_point)
        if self.vehicle is None:
            self.vehicle = self.world.spawn_actor(vehicle_bp, spawn_point)

        # Configure and spawn the camera sensor
        camera_bp = blueprint_library.find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', str(width))
        camera_bp.set_attribute('image_size_y', str(height))
        camera_bp.set_attribute('fov', str(fov))

        camera_transform = carla.Transform(carla.Location(x=1.5, z=1.6))
        self.camera: carla.Sensor = self.world.spawn_actor(camera_bp, camera_transform, attach_to=self.vehicle)

        # Start listening to camera frames
        self.camera.listen(self._on_image)
        self.get_logger().info(
            f'Carla camera started on {host}:{port}, publishing {image_topic} and {info_topic} ({width}x{height}, fov={fov}).'
        )

    def _on_image(self, image: 'carla.Image') -> None:
        # CARLA gives BGRA uint8
        np_img = np.frombuffer(image.raw_data, dtype=np.uint8).reshape((image.height, image.width, 4))
        bgr = np_img[:, :, :3]

        now = self.get_clock().now().to_msg()
        image_msg = Image()
        image_msg.header.stamp = now
        image_msg.header.frame_id = 'camera_front'
        image_msg.height = image.height
        image_msg.width = image.width
        image_msg.encoding = 'bgr8'
        image_msg.is_bigendian = 0
        image_msg.step = image.width * 3
        image_msg.data = bgr.tobytes()
        self.publisher_image.publish(image_msg)

        info_msg = CameraInfo()
        info_msg.header = image_msg.header
        info_msg.width = image.width
        info_msg.height = image.height
        focal_length = image.width / (2.0 * math.tan(math.radians(self.fov) / 2.0))
        info_msg.k = [
            focal_length, 0.0, image.width / 2.0,
            0.0, focal_length, image.height / 2.0,
            0.0, 0.0, 1.0,
        ]
        info_msg.p = [
            focal_length, 0.0, image.width / 2.0, 0.0,
            0.0, focal_length, image.height / 2.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
        ]
        self.publisher_info.publish(info_msg)

    def destroy(self) -> None:
        try:
            if getattr(self, 'camera', None) is not None:
                self.camera.stop()
                self.camera.destroy()
            if getattr(self, 'vehicle', None) is not None:
                self.vehicle.destroy()
        finally:
            super().destroy_node()


def main() -> None:
    rclpy.init()
    node = CarlaCameraNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
