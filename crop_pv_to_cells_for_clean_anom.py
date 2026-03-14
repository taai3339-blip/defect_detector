import cv2
import os
from ultralytics import YOLO
import numpy as np
from collections import Counter


CELL_PADDING = 10
IMAGE_SIZE = 1280
CELL_MODEL_PATH = r"models\cell_detector_best_model_8s_7_1.pt"

# cell_model = YOLO(CELL_DETECTOR_MODEL)
cell_model = YOLO(CELL_MODEL_PATH)

train_img_dir = r"C:\Users\Rowan\Documents\Rowan\Yolo_test\cropped_clean_pvs"
out_img_dir = r"C:\Users\Rowan\Documents\Rowan\Yolo_test\anomaly\cropped_clean_using_yo8s_1280_p10"
os.makedirs(out_img_dir, exist_ok=True)

# Function to compute mode cell width
def compute_cell_stats(results):
    widths = []
    for det in results[0].boxes.xyxy:
        x_min, y_min, x_max, y_max = map(int, det)
        widths.append(x_max - x_min)
    if not widths:
        return 0, []  # fallback
    # Mode width
    width_counts = Counter(widths)
    mode_width = width_counts.most_common(1)[0][0]
    # Return mode and all widths for stats
    return mode_width, widths

def compute_mode_width(boxes):
    widths = [(int(x2)-int(x1)) for x1, y1, x2, y2 in boxes]
    if not widths:
        return 0, widths
    width_counts = Counter(widths)
    mode_width = width_counts.most_common(1)[0][0]
    return mode_width, widths

# --- Filter out outlier boxes ---
def filter_boxes_by_mode(boxes, tolerance=0.05):
    """
    boxes: list of [x_min, y_min, x_max, y_max]
    tolerance: fraction deviation from mode (e.g., 0.3 → ±30%)
    """
    mode_width, widths = compute_mode_width(boxes)
    filtered_boxes = []
    
    for box in boxes:
        x_min, y_min, x_max, y_max = map(int, box)
        width = x_max - x_min
        if abs(width - mode_width) / mode_width <= tolerance:
            filtered_boxes.append(box)
    
    return filtered_boxes


def sort_boxes_top_down_left_right(boxes, y_tolerance_ratio=0.5):
    """
    boxes: list of [x_min, y_min, x_max, y_max]
    y_tolerance_ratio: fraction of average box height to group rows
    """

    if not boxes:
        return []

    # Compute average height (for row grouping)
    heights = [(b[3] - b[1]) for b in boxes]
    avg_height = sum(heights) / len(heights)
    y_tol = avg_height * y_tolerance_ratio

    # Sort by y_min first
    boxes = sorted(boxes, key=lambda b: b[1])

    rows = []
    for box in boxes:
        placed = False
        for row in rows:
            # Compare y_min with first box in row
            if abs(box[1] - row[0][1]) <= y_tol:
                row.append(box)
                placed = True
                break
        if not placed:
            rows.append([box])

    # Sort each row left → right
    for row in rows:
        row.sort(key=lambda b: b[0])

    # Flatten rows
    sorted_boxes = [box for row in rows for box in row]

    return sorted_boxes




# ----- Main workflow -----
def main():
    image_files = sorted(os.listdir(train_img_dir))

    for img_name in image_files:
        img_path = os.path.join(train_img_dir, img_name)

        if os.path.isdir(img_path):
            continue
        
        pv_img = cv2.imread(img_path)
        pv_h, pv_w = pv_img.shape[:2]
        
        pad = CELL_PADDING

        # Run cell detection
        results = cell_model.predict(pv_img, conf=0.4, iou=0.6, imgsz=IMAGE_SIZE, verbose=False)
        
        # Compute cell stats
        mode_width, all_widths = compute_cell_stats(results)

        if len(all_widths) == 0:
            print("no widths. continue", img_name)
            continue
        
        # Define abnormal thresholds
        too_wide_thresh = mode_width * 1.5
        too_narrow_thresh = mode_width * 0.5

        # Count abnormal cells
        num_wide = sum(1 for w in all_widths if w > too_wide_thresh)
        num_narrow = sum(1 for w in all_widths if w < too_narrow_thresh)

        # Decision logic
        if num_wide > len(all_widths) * 0.2:  # >20% cells are too wide
            print("Too many wide cells → rerunning preprocessing with lower threshold", img_name)
            # continue
            # Run cell detection
            results = cell_model.predict(pv_img, conf=0.45, iou=0.6,  imgsz=IMAGE_SIZE)

            # Compute cell stats
            mode_width, all_widths = compute_cell_stats(results)


        elif num_narrow > len(all_widths) * 0.2:  # >20% too narrow
            print("Many small cells → pass as is", img_name)
        # else:
        #     print("Cell sizes OK → proceed")
        

        # --- Usage with YOLO results ---
        boxes = results[0].boxes.xyxy.cpu().numpy()  # list of detected boxes
        boxes = [list(b) for b in boxes]  # convert to Python list if needed

        filtered_boxes = filter_boxes_by_mode(boxes)        

        sorted_boxes = sort_boxes_top_down_left_right(filtered_boxes)


        for i, det in enumerate(sorted_boxes):
            x1, y1, x2, y2 = map(int, det)
            
            # cv2.putText(full_img_viz, str(i), (x1, y1 - 10), 
            #             cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2)
            
            # Calculate padded coordinates, clamping to image boundaries
            x1_padded = max(0, x1 - pad)
            y1_padded = max(0, y1 - pad)
            x2_padded = min(pv_w, x2 + pad)
            y2_padded = min(pv_h, y2 + pad)
            
            # Crop with padding
            cell_crop = pv_img[y1_padded:y2_padded, x1_padded:x2_padded]
            if cell_crop.size == 0:
                continue
            
            # Original cell dimensions (without padding) in the padded crop
            crop_w = x2_padded - x1_padded
            crop_h = y2_padded - y1_padded
            
            # Calculate padding offset in the actual crop (may be less than pad if near image edge)
            pad_x = x1 - x1_padded
            pad_y = y1 - y1_padded
            
            cell_img_name = f"{os.path.splitext(img_name)[0]}_cell{i}.jpg"

            cv2.imwrite(os.path.join(out_img_dir, cell_img_name), cell_crop)


if __name__ == "__main__":
    main()
