from ultralytics import YOLO

# Load a model
model = YOLO("yolo11n-seg.yaml")

# Train the model
results = model.train(data="/home/jasperxzy/dataset/CULane_yolo_seg/data.yaml", 
                      epochs=100, 
                      imgsz=640,
                      batch=64,
                      cache=True,
                      workers=16,
                      project="yolo11n-seg",
                      name="yolo11n-lane"
                      )
