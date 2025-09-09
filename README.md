# AuraCar

## carla

### carla_ue5

- On-screen Rendering
```bash
./CarlaUnreal.sh -quality-level=Epic
```

- Off-screen Rendering
```bash
./CarlaUnreal.sh -quality-level=Epic -RenderOffScreen
```

### carla_bringup

```bash
ros2 launch carla_bringup multi_vehicle_bringup.launch.py config:=./config/multi_vehicle.yaml
```

### carla_teleop_keyboard

```bash
ros2 launch carla_teleop_keyboard teleop_keyboard.launch.py vehicle_id:=ego mode:=ackermann
```

## foxglove

### foxglove_bridge

```bash
ros2 launch foxglove_bridge foxglove_bridge_launch.xml
```
