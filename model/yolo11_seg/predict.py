import cv2
import numpy as np
from ultralytics import YOLO
import random
import os
import torch

def yolo11_seg_inference_video(
    video_path: str,
    model_path: str = 'yolov8n-seg.pt',
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.7,
    mask_alpha: float = 0.5,
    show_fps: bool = True
):
    # 载入 YOLO11 分割模型
    try:
        model = YOLO(model_path)
        print(f"模型 {model_path} 载入成功。")
    except Exception as e:
        print(f"载入模型失败: {e}")
        return

    # 尝试使用 GPU，如果不可用则使用 CPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Running on device: {device}")
    model.to(device)

    # 打开视频文件
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"错误: 无法打开视频文件 {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"视频信息: FPS={fps}, 尺寸={frame_width}x{frame_height}")

    prev_frame_time = 0
    new_frame_time = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("视频结束或无法读取帧，退出循环。")
            break

        display_frame = frame.copy() 

        results = model.predict(
            display_frame,
            conf=conf_threshold,
            iou=iou_threshold,
            device=device,
            stream=True, 
            verbose=False 
        )

        mask_layer = np.zeros_like(display_frame, dtype=np.uint8)

        for r in results: 
            if r.masks is not None:
                for i, mask in enumerate(r.masks.data):
                    color_bgr = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                    
                    mask_np = mask.cpu().numpy()
                    mask_resized = cv2.resize(mask_np, (frame_width, frame_height), interpolation=cv2.INTER_NEAREST)
                    true_mask = mask_resized > 0.5 
                    
                    mask_layer[true_mask] = color_bgr
            
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = box.conf[0]
                cls_id = int(box.cls[0])
                label = f"{model.names[cls_id]} {conf:.2f}"

                cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                cv2.rectangle(display_frame, (x1, y1 - text_height - baseline), (x1 + text_width, y1), (0, 255, 0), -1)
                cv2.putText(display_frame, label, (x1, y1 - baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        
        display_frame = cv2.addWeighted(display_frame, 1, mask_layer, mask_alpha, 0)
        
        if show_fps:
            new_frame_time = cv2.getTickCount()
            fps = cv2.getTickFrequency() / (new_frame_time - prev_frame_time)
            prev_frame_time = new_frame_time
            fps_text = f"FPS: {int(fps)}"
            cv2.putText(display_frame, fps_text, (frame_width - 150, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            
        cv2.imshow('YOLO11 Segmentation Inference', display_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("视频处理完成，资源已释放。")

# --- 使用示例 ---
if __name__ == "__main__":
    video_path = '/home/jasperxzy/dataset/CULane_yolo_seg/test.mp4'
    # video_path = 0 # 使用默认摄像头

    model_seg_path = '/home/jasperxzy/projects/AuraCar/model/yolo11_seg/yolo11n-seg/yolo11n-lane/weights/best.pt'

    if not os.path.exists(model_seg_path):
        print(f"错误: 模型文件 '{model_seg_path}' 不存在或无法访问。")
    else:
        yolo11_seg_inference_video(
            video_path=video_path,
            model_path=model_seg_path,
            conf_threshold=0.3,  
            iou_threshold=0.5,   
            mask_alpha=0.6,      
            show_fps=True
        )
