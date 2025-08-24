import os
import cv2
import numpy as np
from tqdm import tqdm
import shutil
import uuid

# 配置参数
CULANE_BASE_DIR = '/home/jasperxzy/dataset/CULane'                  # CULane 数据集的根目录
YOLO_DATA_ROOT = '/home/jasperxzy/dataset/CULane_yolo_seg'          # YOLO11 格式数据集的输出目录

# YOLO11 的类别名称和 ID
CLASSES = {
    'lane': 0 
}

# 从分割掩码中提取 YOLO 格式的多边形
def mask_to_yolo_polygons(mask_path, image_width, image_height, class_id, error_log):
    """
    从灰度语义分割掩码图片中提取每个独立车道线的轮廓，并转换为YOLOv8分割格式。
    Args:
        mask_path (str): 语义分割掩码图片的文件路径。
        image_width (int): 原始图像宽度。
        image_height (int): 原始图像高度。
        class_id (int): YOLO类别ID。
        error_log (list): 用于收集错误和警告信息的列表。
    Returns:
        list: 一个列表，每个元素是代表一个车道实例的YOLO标签字符串。
    """
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        error_log.append(f"Warning: Could not read mask image: {mask_path}. Skipping mask processing.")
        return []

    yolo_polygons = []
    
    unique_values = np.unique(mask)
    lane_instance_ids = [val for val in unique_values if val != 0] # 排除背景0

    if not lane_instance_ids:
        # error_log.append(f"Info: No lane instances found in mask: {mask_path}.") # 对于无车道是正常情况，不报 warning
        return []

    for instance_val in lane_instance_ids:
        binary_lane_mask = (mask == instance_val).astype(np.uint8) * 255
        
        # 寻找轮廓
        # cv2.RETR_EXTERNAL: 只提取最外层轮廓
        # cv2.CHAIN_APPROX_SIMPLE: 压缩水平、垂直和对角线线段，减少点数量
        contours, _ = cv2.findContours(binary_lane_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            if len(contour) < 3:
                # error_log.append(f"Warning: Contour with less than 3 points found in mask {mask_path} for value {instance_val}. Skipping.")
                continue # 轮廓点过少无法形成有效多边形，跳过

            polygon_coords = contour.reshape(-1, 2)
            
            normalized_polygon = []
            for px, py in polygon_coords:
                # 归一化坐标并四舍五入到6位小数
                # 确保坐标在0到1之间，即使 cv2.findContours 可能给出边缘值
                nx = np.clip(px / image_width, 0.0, 1.0)
                ny = np.clip(py / image_height, 0.0, 1.0)
                normalized_polygon.append(f"{nx:.6f}")
                normalized_polygon.append(f"{ny:.6f}")
            
            if len(normalized_polygon) > 0:
                yolo_polygons.append(f"{class_id} {' '.join(normalized_polygon)}")
    
    return yolo_polygons

# 主转换函数
def convert_culane_to_yolo_seg(culane_base_dir, yolo_data_root, classes):
    os.makedirs(yolo_data_root, exist_ok=True)
    
    train_images_dir = os.path.join(yolo_data_root, 'images', 'train')
    train_labels_dir = os.path.join(yolo_data_root, 'labels', 'train')
    val_images_dir = os.path.join(yolo_data_root, 'images', 'val')
    val_labels_dir = os.path.join(yolo_data_root, 'labels', 'val')
    
    os.makedirs(train_images_dir, exist_ok=True)
    os.makedirs(train_labels_dir, exist_ok=True)
    os.makedirs(val_images_dir, exist_ok=True)
    os.makedirs(val_labels_dir, exist_ok=True)

    splits = {
        'train': 'list/train_gt.txt',
        'val': 'list/val_gt.txt'
    }

    class_id = classes['lane']
    
    total_train_images = 0
    total_val_images = 0
    all_warnings = []

    for split_name, split_file_path_rel in splits.items():
        print(f"\nProcessing {split_name} split...")
        split_file_path = os.path.join(culane_base_dir, split_file_path_rel)
        
        current_images_output_dir = train_images_dir if split_name == 'train' else val_images_dir
        current_labels_output_dir = train_labels_dir if split_name == 'train' else val_labels_dir

        try:
            with open(split_file_path, 'r') as f:
                lines = f.readlines()
        except FileNotFoundError:
            all_warnings.append(f"Error: Split file not found: {split_file_path}. Skipping {split_name} split.")
            continue
        except Exception as e:
            all_warnings.append(f"Error reading split file {split_file_path}: {e}. Skipping {split_name} split.")
            continue
            
        num_processed_images_in_split = 0

        # 获取当前 split 的总行数用于 tqdm 的 total 参数
        split_total_lines = len(lines)

        for line_idx, line in enumerate(tqdm(lines, desc=f"Converting {split_name}", total=split_total_lines)):
            parts = line.strip().split()
            
            if len(parts) < 2:
                all_warnings.append(f"Warning: Line {line_idx+1} in {split_file_path} has unexpected format: '{line.strip()}'. Skipping.")
                continue

            image_rel_path = parts[0][1:] 
            gt_mask_rel_path = parts[1][1:]

            original_image_path = os.path.join(culane_base_dir, image_rel_path)
            original_gt_mask_path = os.path.join(culane_base_dir, gt_mask_rel_path)
            
            new_filename_base = str(uuid.uuid4())
            
            original_image_ext = os.path.splitext(original_image_path)[1]
            # original_mask_ext = os.path.splitext(original_gt_mask_path)[1] # mask extension is not needed for target filename

            new_image_filename = f"{new_filename_base}{original_image_ext}"
            new_label_filename = f"{new_filename_base}.txt"

            dest_image_path = os.path.join(current_images_output_dir, new_image_filename)
            
            # 复制图片
            try:
                shutil.copyfile(original_image_path, dest_image_path)
            except FileNotFoundError:
                all_warnings.append(f"Warning: Original image file not found: {original_image_path}. Skipping.")
                continue # 如果原始图片不存在，就直接跳过

            # 读取图片获取尺寸
            img = cv2.imread(original_image_path) # 这里仍然用 original_image_path 读取，因为它更可靠
            if img is None:
                all_warnings.append(f"Warning: Could not read image (corrupted/invalid format): {original_image_path}. Removing copied image and skipping.")
                if os.path.exists(dest_image_path):
                    os.remove(dest_image_path) # 删除已复制的损坏图片
                continue
            h, w, _ = img.shape
            
            # 从掩码文件中提取所有多边形标签
            # 直接传递 all_warnings，让 mask_to_yolo_polygons 函数记录内部问题
            yolo_label_content = mask_to_yolo_polygons(original_gt_mask_path, w, h, class_id, all_warnings) 
            
            # 将标签内容写入文件或删除图片
            if yolo_label_content: # 如果成功提取到有效多边形
                dest_label_path = os.path.join(current_labels_output_dir, new_label_filename)
                with open(dest_label_path, 'w') as out_f:
                    out_f.write("\n".join(yolo_label_content))
                num_processed_images_in_split += 1
            else: # 如果没有有效车道线或掩码处理失败
                all_warnings.append(f"Warning: No valid lane polygons found or mask processing failed for: {original_gt_mask_path}. Removing copied image and skipping.")
                if os.path.exists(dest_image_path):
                    os.remove(dest_image_path) # 删除对应的已复制图片
                # 此时不需要创建空的标签文件，因为图片被删除了

        if split_name == 'train':
            total_train_images = num_processed_images_in_split
        else:
            total_val_images = num_processed_images_in_split

    print(f"\n--- Conversion Summary ---")
    print(f"Total images successfully converted for train split: {total_train_images}")
    print(f"Total images successfully converted for val split: {total_val_images}")

    # 报告所有收集到的警告和错误
    if all_warnings:
        print("\n--- Warnings and Errors Encountered ---")
        for warn in all_warnings:
            print(warn)
    else:
        print("\nNo warnings or errors encountered during conversion.")

    # 创建 data.yaml 文件 
    data_yaml_content = f"""
path: "{os.path.abspath(yolo_data_root)}"
train: images/train
val: images/val

nc: {len(classes)}
names: {list(classes.keys())}
"""
    with open(os.path.join(yolo_data_root, 'data.yaml'), 'w') as f:
        f.write(data_yaml_content)

    print(f"\nCULane to Ultralytics Segmentation (from masks, with UUID renaming) conversion complete!")
    print(f"Dataset saved to: {os.path.abspath(yolo_data_root)}")

if __name__ == "__main__":
    convert_culane_to_yolo_seg(CULANE_BASE_DIR, YOLO_DATA_ROOT, CLASSES)
