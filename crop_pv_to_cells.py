import cv2
import os
from ultralytics import YOLO
import numpy as np
from collections import Counter


# train_img_dir = r"C:\Users\Rowan\Documents\Rowan\Yolo_test\modified_mixed_pv_orig_res\test\images"
# train_lbl_dir = r"C:\Users\Rowan\Documents\Rowan\Yolo_test\modified_mixed_pv_orig_res\test\labels" 
train_img_dir = r"C:\Users\Rowan\Documents\Rowan\Yolo_test\Mixed-PV---Original-Res-1\test\images" # made on modified_mixed_pv_orig_res_more_classes
train_lbl_dir = r"C:\Users\Rowan\Documents\Rowan\Yolo_test\Mixed-PV---Original-Res-1\test\labels"

out_img_dir = r'C:\Users\Rowan\Documents\Rowan\Yolo_test\pv_mixed_crop_orig_res_before_merge\test\images'
out_lbl_dir = r'C:\Users\Rowan\Documents\Rowan\Yolo_test\pv_mixed_crop_orig_res_before_merge\test\labels'

CELL_PADDING = 4
IMAGE_SIZE = 1280
CELL_MODEL_PATH = r"models\cell_detector_best_model_8s_7_1.pt"

# cell_model = YOLO(CELL_DETECTOR_MODEL)
cell_model = YOLO(CELL_MODEL_PATH)

os.makedirs(out_img_dir, exist_ok=True)
os.makedirs(out_lbl_dir, exist_ok=True)



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


def find_best_pv_contour(contours, image_width, image_height):
    """
    Filters through many contours to find the one that looks most like a PV module.
    Returns the best contour or None.
    """
    
    # 1. Define Thresholds (Relative to Image Size)
    # A PV module usually occupies between 10% and 90% of the image area.
    # This filters out small specs (1%) or the whole white background (100%).
    image_area = image_width * image_height
    min_area = 0.05 * image_area  # Must be at least 5% of the image
    max_area = 0.95 * image_area  # Cannot be 95%+ (likely a failed threshold)

    best_contour = None
    best_score = 0

    for cnt in contours:
        area = cv2.contourArea(cnt)

        # FILTER 1: Ignore tiny noise
        if area < 1.0:
            continue
        if area < 0.01 * image_area or  area > 0.98 * image_area:
            continue

        # FILTER 2: Aspect Ratio Check (Geometry)
        # Get the bounding rectangle
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = float(w) / h

        # PV modules are rarely extremely thin lines or perfect circles.
        # Typical PV ratios are between 1:2 and 2:1 (0.5 to 2.0).
        # We use a wider range (0.3 to 3.0) to be safe for different camera angles.
        if aspect_ratio < 0.3 or aspect_ratio > 3.5:
            continue

        # FILTER 3: Solidity Check (Shape Integrity)
        # Solidity = Contour Area / Convex Hull Area
        # A perfect rectangle has solidity of 1.0.
        # Broken frames or irregular blobs have solidity < 0.8.
        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        
        if hull_area == 0: continue # Avoid division by zero
        
        solidity = float(area) / hull_area
        
        if solidity < 0.8:
            continue # This contour is likely jagged or broken, skip it.

        # --- SCORING ---
        # If it passed all filters, we need to pick the "Best" one.
        # Usually, the "Real PV" is the largest valid contour.
        # We can use Area directly as the score.
        if area > best_score:
            best_score = area
            best_contour = cnt

    return best_contour


def extract_cell_labels_simple(
    pv_label_file,
    cell_bbox,      # [x_min, y_min, x_max, y_max] in ORIGINAL pixels
    pv_w,           # Original width
    pv_h,           # Original height
    pad_x,          # Padding offset in crop
    pad_y,          # Padding offset in crop
    crop_w,         # Width of the crop
    crop_h,         # Height of the crop
    min_overlap_ratio=0.1
):
    x_min, y_min, x_max, y_max = cell_bbox
    
    cell_labels = []

    if not os.path.exists(pv_label_file):
        return cell_labels

    with open(pv_label_file, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
                
            cls = int(float(parts[0]))
            xc, yc, bw, bh = map(float, parts[1:5])

            # 1. Convert normalized defect to absolute pixels (Original Space)
            defect_x1 = (xc - bw / 2) * pv_w
            defect_y1 = (yc - bh / 2) * pv_h
            defect_x2 = (xc + bw / 2) * pv_w
            defect_y2 = (yc + bh / 2) * pv_h
            
            defect_area = (defect_x2 - defect_x1) * (defect_y2 - defect_y1)
            
            # 2. Calculate Intersection (All in Original Space)
            inter_x1 = max(defect_x1, x_min)
            inter_y1 = max(defect_y1, y_min)
            inter_x2 = min(defect_x2, x_max)
            inter_y2 = min(defect_y2, y_max)
            
            inter_w = max(0, inter_x2 - inter_x1)
            inter_h = max(0, inter_y2 - inter_y1)
            inter_area = inter_w * inter_h
            
            # Check overlap
            if defect_area > 0 and inter_area / defect_area >= min_overlap_ratio:
                
                # 3. Map to Crop Coordinates
                # The crop starts at (x_min - pad_x, y_min - pad_y) in the Original Image
                # We need to find where the defect is RELATIVE to the crop start
                
                crop_start_x = x_min - pad_x
                crop_start_y = y_min - pad_y
                
                # Defect coords relative to crop top-left
                rel_x1 = inter_x1 - crop_start_x
                rel_y1 = inter_y1 - crop_start_y
                rel_x2 = inter_x2 - crop_start_x
                rel_y2 = inter_y2 - crop_start_y
                
                # 4. Normalize to Crop Dimensions
                cx = ((rel_x1 + rel_x2) / 2) / crop_w
                cy = ((rel_y1 + rel_y2) / 2) / crop_h
                cw = (rel_x2 - rel_x1) / crop_w
                ch = (rel_y2 - rel_y1) / crop_h
                
                # Sanity check
                if cw > 0.01 and ch > 0.01:
                    cx = max(0, min(1, cx))
                    cy = max(0, min(1, cy))
                    cw = max(0, min(1, cw))
                    ch = max(0, min(1, ch))
                    
                    cell_labels.append(f"{cls} {cx:.6f} {cy:.6f} {cw:.6f} {ch:.6f}")

    return cell_labels


# ----- Main workflow -----
def main():
    failed_images = []
    image_files = sorted(os.listdir(train_img_dir))

    for img_name in image_files:
        img_path = os.path.join(train_img_dir, img_name)
        lbl_path = os.path.join(
            train_lbl_dir,
            os.path.splitext(img_name)[0] + ".txt"
        )

        pv_img = cv2.imread(img_path)
        pv_h, pv_w = pv_img.shape[:2]
        
        pad = CELL_PADDING

        # Run cell detection
        results = cell_model.predict(pv_img, conf=0.45, iou=0.6, imgsz=IMAGE_SIZE, verbose=False)
        
        # Compute cell stats
        mode_width, all_widths = compute_cell_stats(results)

        if len(all_widths) == 0:
            print("no widths. continue", img_name)
            failed_images.append(img_path)
            continue
        elif len(all_widths) <= 10: # udate made later not applied in previous run
            print("detecting low amount than normal", img_name)
            failed_images.append(img_path)
            continue

        # Define abnormal thresholds
        too_wide_thresh = mode_width * 1.5
        too_narrow_thresh = mode_width * 0.5

        # Count abnormal cells
        num_wide = sum(1 for w in all_widths if w > too_wide_thresh)
        num_narrow = sum(1 for w in all_widths if w < too_narrow_thresh)

        # Decision logic
        if num_wide > len(all_widths) * 0.2:  # >20% cells are too wide
            print("Too many wide cells", img_name)
            # continue

        elif num_narrow > len(all_widths) * 0.2:  # >20% too narrow
            print("Many small cells → pass as is", img_name)
        # else:
        #     print("Cell sizes OK → proceed")
        

        # --- Usage with YOLO results ---
        boxes = results[0].boxes.xyxy.cpu().numpy()  # list of detected boxes
        boxes = [list(b) for b in boxes]  # convert to Python list if needed

        filtered_boxes = filter_boxes_by_mode(boxes)        

        sorted_boxes = sort_boxes_top_down_left_right(filtered_boxes)

        # full_img_viz = pv_img.copy()
        # orig_h, orig_w = pv_img.shape[:2]

        # if os.path.exists(lbl_path):
        #     with open(lbl_path, "r") as f:
        #         for line in f:
        #             parts = line.strip().split()
        #             if len(parts) < 5:
        #                 continue
        #             # Parse PV label (normalized 0-1)
        #             _, xc, yc, w, h = map(float, parts)
                    
        #             # DIRECT conversion to pixels (Original Space)
        #             # Because pv_w should equal orig_w in this workflow
        #             defect_xc_px = xc * orig_w
        #             defect_yc_px = yc * orig_h
        #             defect_w_px  = w  * orig_w
        #             defect_h_px  = h  * orig_h

        #             x1_def = int(defect_xc_px - defect_w_px / 2)
        #             y1_def = int(defect_yc_px - defect_h_px / 2)
        #             x2_def = int(defect_xc_px + defect_w_px / 2)
        #             y2_def = int(defect_yc_px + defect_h_px / 2)
                    
        #             # Draw original defects in RED
        #             cv2.rectangle(full_img_viz, (x1_def, y1_def), (x2_def, y2_def), (0, 0, 255), 2)


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
            cell_lbl_name = f"{os.path.splitext(img_name)[0]}_cell{i}.txt"

            cv2.imwrite(os.path.join(out_img_dir, cell_img_name), cell_crop)

    
            cell_labels = extract_cell_labels_simple(
                lbl_path,           # Path to original PV labels
                det,                # [x1, y1, x2, y2] in ORIGINAL pixels
                pv_w, pv_h,     # Original image dimensions
                pad_x, pad_y,       # Offsets within the crop
                crop_w, crop_h      # Crop dimensions
            )
            
            cell_lbl_file = os.path.join(out_lbl_dir, cell_lbl_name)

            if len(cell_labels) > 0:
                with open(cell_lbl_file, "w") as f:
                    f.write("\n".join(cell_labels))

            # left_side = full_img_viz.copy()
            # # Draw the cell boundary in GREEN (Thick line)
            # cv2.rectangle(left_side, (x1, y1), (x2, y2), (0, 255, 0), 3)

            # right_side = cell_crop.copy()
            # for label_str in cell_labels:
            #     parts = label_str.split()
            #     # coords are normalized to crop_w/crop_h
            #     xc, yc, w_norm, h_norm = map(float, parts[1:5])
                
            #     # Convert to pixels relative to crop
            #     x_c = int(xc * crop_w)
            #     y_c = int(yc * crop_h)
            #     w_px = int(w_norm * crop_w)
            #     h_px = int(h_norm * crop_h)
                
            #     x1_c = x_c - w_px // 2
            #     y1_c = y_c - h_px // 2
            #     x2_c = x_c + w_px // 2
            #     y2_c = y_c + h_px // 2
                
            #     # Draw defect in RED
            #     cv2.rectangle(right_side, (x1_c, y1_c), (x2_c, y2_c), (0, 0, 255), 2)

            # # 3. Combine Images
            # # We need to resize the right_side (crop) to match the height of left_side (full image)
            # # so they are easy to compare side-by-side.
            # target_h, target_w = left_side.shape[:2]
            # crop_h_curr, crop_w_curr = right_side.shape[:2]
            
            # # Calculate new width to maintain aspect ratio
            # new_w = int(crop_w_curr * (target_h / crop_h_curr))
            # right_side_resized = cv2.resize(right_side, (new_w, target_h))
            
            # # Concatenate horizontally
            # # Add some black padding in between if desired, or just hstack
            # comparison_img = np.hstack((left_side, right_side_resized))
            
            # # Save comparison image
            # debug_img_name = f"{os.path.splitext(img_name)[0]}_cell{i}_compare.jpg"
            # cv2.imwrite(os.path.join(out_img_dir, 'check', debug_img_name), comparison_img)

    with open(os.path.join(out_img_dir, "failed_pv.txt"), 'w') as f:
        f.writelines(f"{image}\n" for image in failed_images)

if __name__ == "__main__":
    main()
