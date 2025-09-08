import os
import time
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist

try:
	from ackermann_msgs.msg import AckermannDriveStamped
except Exception:
	AckermannDriveStamped = None

try:
	import pygame
except Exception as exc:
	raise SystemExit('pygame not installed. Please install with: pip install pygame') from exc


class TeleopKeyboardNode(Node):
	def __init__(self) -> None:
		super().__init__('carla_teleop_keyboard_pygame')

		# 车辆命名空间与控制模式
		self.vehicle_id = self.declare_parameter('vehicle_id', 'ego').get_parameter_value().string_value
		self.mode = self.declare_parameter('mode', 'twist').get_parameter_value().string_value  				# twist|ackermann

		# 运行频率/上限/变化率
		self.rate_hz = int(self.declare_parameter('rate', 60).get_parameter_value().integer_value)
		self.max_speed = float(self.declare_parameter('max_speed', 12.0).get_parameter_value().double_value)  	# m/s
		self.max_steer = float(self.declare_parameter('max_steer', 0.7).get_parameter_value().double_value)  	# rad
		self.accel_rate = float(self.declare_parameter('accel_rate', 3.0).get_parameter_value().double_value)  	# m/s^2
		self.brake_rate = float(self.declare_parameter('brake_rate', 6.0).get_parameter_value().double_value)  	# m/s^2
		self.steer_rate = float(self.declare_parameter('steer_rate', 1.8).get_parameter_value().double_value)  	# rad/s
		self.echo_keys = bool(self.declare_parameter('echo_keys', False).get_parameter_value().bool_value)

		# 创建小窗口抓取键盘焦点，并显示按键 HUD
		pygame.init()
		pygame.display.set_caption('Teleop')
		self.screen = pygame.display.set_mode((240, 120))
		self.font = pygame.font.SysFont(None, 36)
		pygame.event.set_allowed([pygame.KEYDOWN, pygame.KEYUP, pygame.QUIT])
		self.clock = pygame.time.Clock()

		ns = f'/vehicle_{self.vehicle_id}'
		if self.mode == 'ackermann' and AckermannDriveStamped is not None:
			self.pub_ack = self.create_publisher(AckermannDriveStamped, f'{ns}/cmd_ackermann', 10)
			self.pub_twist = None
		else:
			self.pub_twist = self.create_publisher(Twist, f'{ns}/cmd_vel', 10)
			self.pub_ack = None

		self.speed = 0.0
		self.steer = 0.0
		self.last_time = time.monotonic()
		self.timer = self.create_timer(1.0 / max(1, self.rate_hz), self._on_timer)

		self.get_logger().info(
			f"Pygame teleop started. mode={self.mode}, vehicle_id={self.vehicle_id}"
		)

	def _on_timer(self) -> None:
		# 轮询事件和按键状态，支持组合键
		events = pygame.event.get()
		for ev in events:
			if ev.type == pygame.QUIT:
				rclpy.shutdown()
				return
		keys = pygame.key.get_pressed()

		now = time.monotonic()
		dt = max(0.0, min(0.1, now - self.last_time))
		self.last_time = now

		# 油门/刹车：按时间积分
		if keys[pygame.K_w] or keys[pygame.K_UP]:
			self.speed = min(self.max_speed, self.speed + self.accel_rate * dt)
		elif keys[pygame.K_s] or keys[pygame.K_DOWN]:
			self.speed = max(-self.max_speed, self.speed - self.brake_rate * dt)
		else:
			# 无输入时自然衰减到 0
			if self.speed > 0.0:
				self.speed = max(0.0, self.speed - self.brake_rate * 0.5 * dt)
			elif self.speed < 0.0:
				self.speed = min(0.0, self.speed + self.brake_rate * 0.5 * dt)

		# 转向：按时间积分，松开回正
		if keys[pygame.K_a] or keys[pygame.K_LEFT]:
			self.steer = max(-self.max_steer, self.steer - self.steer_rate * dt)
		elif keys[pygame.K_d] or keys[pygame.K_RIGHT]:
			self.steer = min(self.max_steer, self.steer + self.steer_rate * dt)
		else:
			# 回正
			if self.steer > 0.0:
				self.steer = max(0.0, self.steer - self.steer_rate * dt)
			elif self.steer < 0.0:
				self.steer = min(0.0, self.steer + self.steer_rate * dt)

		# 急停
		if keys[pygame.K_q]:
			self.speed = 0.0
			self.steer = 0.0

		# 绘制按键 HUD
		self.screen.fill((30, 30, 30))
		pressed = []
		if keys[pygame.K_w] or keys[pygame.K_UP]:
			pressed.append('W')
		if keys[pygame.K_s] or keys[pygame.K_DOWN]:
			pressed.append('S')
		if keys[pygame.K_a] or keys[pygame.K_LEFT]:
			pressed.append('A')
		if keys[pygame.K_d] or keys[pygame.K_RIGHT]:
			pressed.append('D')
		if keys[pygame.K_q]:
			pressed.append('Q')
		text = self.font.render(' '.join(pressed) if pressed else '-', True, (200, 200, 200))
		self.screen.blit(text, (20, 40))
		pygame.display.flip()

		# 发布控制消息
		if self.pub_twist is not None:
			msg = Twist()
			msg.linear.x = float(self.speed)
			msg.angular.z = float(self.steer)
			self.pub_twist.publish(msg)
		elif self.pub_ack is not None and AckermannDriveStamped is not None:
			msg = AckermannDriveStamped()
			msg.drive.speed = float(self.speed)
			msg.drive.steering_angle = float(self.steer)
			self.pub_ack.publish(msg)


def main() -> None:
	rclpy.init()
	node = TeleopKeyboardNode()
	try:
		rclpy.spin(node)
	except KeyboardInterrupt:
		pass
	finally:
		node.destroy_node()
		rclpy.shutdown()


if __name__ == '__main__':
	main()
