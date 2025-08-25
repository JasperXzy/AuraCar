from ultralytics import YOLO

model = YOLO("/home/jasperxzy/projects/AuraCar/model/yolo11_seg/yolo11n-seg/yolo11n-lane/weights/best.pt")

model.export(format="onnx")
