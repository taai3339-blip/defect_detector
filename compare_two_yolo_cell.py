import cv2
import numpy as np
from ultralytics import YOLO
import os

# ---------------------------------------------------------
# HELPER FUNCTIONS
# ---------------------------------------------------------

def calculate_iou(box1, box2):
    """
    Calculate Intersection over Union (IoU) for two bounding boxes.
    Boxes must be in format [x1, y1, x2, y2] (pixel coordinates).
    """
    # Determine the coordinates of the intersection rectangle
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # Calculate area of intersection rectangle
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # Calculate area of both bounding boxes
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    # Calculate union
    union_area = box1_area + box2_area - intersection_area

    # Avoid division by zero
    if union_area == 0:
        return 0.0

    return intersection_area / union_area

def yolo_to_pixels(box, img_w, img_h):
    """
    Convert YOLO normalized format [class, x_center, y_center, width, height]
    to pixel format [x1, y1, x2, y2].
    """
    x_c, y_c, w, h = box
    x1 = int((x_c - w / 2) * img_w)
    y1 = int((y_c - h / 2) * img_h)
    x2 = int((x_c + w / 2) * img_w)
    y2 = int((y_c + h / 2) * img_h)
    return [x1, y1, x2, y2]

# ---------------------------------------------------------
# MAIN COMPARISON LOGIC
# ---------------------------------------------------------

def compare_models(image_path, label_path, model_paths, conf_threshold=0.5, iou_threshold=0.5):
    
    # 1. Load Image
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        return
    img = cv2.imread(image_path)
    h_img, w_img = img.shape[:2]

    # 2. Parse Ground Truth (YOLO format: class x_center y_center w h)
    gt_boxes = []
    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                class_id = int(parts[0])
                coords = list(map(float, parts[1:]))
                pixel_box = yolo_to_pixels(coords, w_img, h_img)
                gt_boxes.append({'class': class_id, 'box': pixel_box})
    else:
        print(f"Warning: Label file not found at {label_path}")

    print(f"Ground Truth Boxes found: {len(gt_boxes)}")
    print("-" * 50)

    # 3. Iterate and Evaluate each model
    for i, model_path in enumerate(model_paths):
        model_name = os.path.basename(model_path)
        print(f"Processing {model_name}...")
        
        # Load model and run inference
        model = YOLO(model_path)
        results = model(img, verbose=False, project="compare_final_cell_defect", save=True, name=str(i)+model_name)
        
        # Extract predictions from results
        # results[0].boxes.data contains [x1, y1, x2, y2, conf, class]
        pred_boxes = []
        for r in results:
            for box in r.boxes:
                # Filter by confidence
                if box.conf[0] >= conf_threshold:
                    pred_boxes.append({
                        'box': box.xyxy[0].tolist(), # [x1, y1, x2, y2]
                        'class': int(box.cls[0]),
                        'conf': float(box.conf[0])
                    })

        # 4. Calculate Metrics (TP, FP, FN)
        tp = 0
        fp = 0
        matched_gt_indices = set()

        # Match predictions to Ground Truth
        for pred in pred_boxes:
            best_iou = 0
            best_gt_idx = -1
            
            for i, gt in enumerate(gt_boxes):
                if i in matched_gt_indices:
                    continue # This GT is already matched
                
                # Check class match
                if pred['class'] != gt['class']:
                    continue

                # Calculate IoU
                iou = calculate_iou(pred['box'], gt['box'])
                
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = i
            
            # Determine if TP or FP
            if best_iou >= iou_threshold:
                tp += 1
                matched_gt_indices.add(best_gt_idx)
            else:
                fp += 1

        fn = len(gt_boxes) - len(matched_gt_indices)

        # 5. Compute Precision, Recall, F1
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        print(f"  Preds: {len(pred_boxes)} (Conf > {conf_threshold})")
        print(f"  TP: {tp}, FP: {fp}, FN: {fn}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  F1 Score:  {f1:.4f}")
        print("-" * 50)

# ---------------------------------------------------------
# USAGE EXAMPLE
# ---------------------------------------------------------
def visualize_and_save(image_path, label_path, model_path, output_filename="debug_comparison5.jpg"):
    # Check if image exists
    if not os.path.exists(image_path):
        print(f"Error: Image file not found at {image_path}")
        return

    # Load Image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read image {image_path}")
        return
        
    h_img, w_img = img.shape[:2]
    print(f"Image loaded: {w_img}x{h_img}")
    
    # 1. DRAW GROUND TRUTH (GREEN)
    gt_count = 0
    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5: continue # Skip malformed lines
                
                class_id = int(parts[0])
                coords = list(map(float, parts[1:]))
            
                box = yolo_to_pixels(coords, w_img, h_img) 
                # box = [int(coords[0]), int(coords[1]), int(coords[2]), int(coords[3])] # FOR PIXEL LABELS
                
                cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
                cv2.putText(img, f"GT {class_id}", (box[0], box[1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
                gt_count += 1
    else:
        print(f"Warning: Label file not found at {label_path}")

    # 2. DRAW PREDICTIONS (RED)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    img_clahe = clahe.apply(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    
    img_bgr = cv2.cvtColor(img_clahe, cv2.COLOR_GRAY2BGR)
        
    model = YOLO(model_path)
    results = model(img_bgr, verbose=False) # fro model 123
    
    pred_count = 0
    for r in results:
        for box in r.boxes:
            # Get coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            
            # Only draw if confidence is high enough
            if conf > 0.5:
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(img, f"Pred {cls} {conf:.2f}", (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
                pred_count += 1

    # 3. SAVE TO FILE
    cv2.imwrite(output_filename, img)


if __name__ == "__main__":
    # Define paths
    base_name = "2_jpg.rf.d889573b107487bcb1188ea5921dfaa9_cell44"
    img_file = r"C:\Users\Rowan\Documents\Rowan\Yolo_test\pv_cropped_ready_integ\test\images\\" + base_name + ".jpg"
    lbl_file = r"C:\Users\Rowan\Documents\Rowan\Yolo_test\pv_cropped_ready_integ\test\labels\\"   + base_name + ".txt" # Standard YOLO label file
    
    # List of models to compare
    models_to_test = [
        r"runs\detect\train97\weights\best.pt",  # Example model 1 . best
        r"runs\detect\train100\weights\best.pt",  # Example model 1 . somehow good
        r"runs\detect\train101\weights\best.pt",  # Example model 1 . bad
        r"runs\detect\train114\weights\best.pt",  # Example model 1 . bad
        r"runs\detect\train115\weights\best.pt",  # Example model 1 . bad
        r"runs\detect\train123\weights\best.pt",  # Example model 1 . 
    ]
    
    # Run comparison
    # (Make sure test_image.jpg and test_image.txt exist in your directory)
    compare_models(img_file, lbl_file, models_to_test, conf_threshold=0.25, iou_threshold=0.5)

import os
# Usage
visualize_and_save(img_file, lbl_file, models_to_test[4], output_filename="debug_comparison5.jpg")