import cv2
import os
from ultralytics import YOLO
import cv2
import numpy as np
from collections import Counter

# Paths
pv_images_path = r"modified_data_1/valid/images"
pv_labels_path = r"modified_data_1/valid/labels"  # YOLO-format PV-level annotations
pv_images_path = r"modified_data_1/valid/images"
pv_labels_path = r"modified_data_1/valid/labels"  # YOLO-format PV-level annotations

CELL_DETECTOR_MODEL = r"C:\Users\Rowan\Documents\Final_demo\Final_demo\demo\demo\demo\demo\src\main\backend\models/best2.pt"  # YOLO trained to detect cells
# CELL_DETECTOR_MODEL = r"runs\detect\train9\weights\best.pt"  # YOLO trained to detect cells
# CELL_DETECTOR_MODEL = r"models\cell_detector_imgs1024.ptt"
CELL_DETECTOR_MODEL = r"runs\detect\train3\weights\best.pt"

IMAGE_SIZE = 960


output_defect_path = "output_cells_defect"
output_clean_path = "output_cells_clean"
base_file_name = "B085_png.rf.ee886bfe6746351a81363739a658b120"
base_file_name = "C209AABB3A20190033_png.rf.24e6c7a051a024707e10d13b194e5c11"

# Example: visualize one PV image with cell numbers
img_file = os.listdir(pv_images_path)[0]  # first PV image
pv_img_path = os.path.join(pv_images_path, img_file)

pv_img_path = r"C:\Users\Rowan\Documents\Rowan\Yolo_test\ELDDS1400c5-dataset-3\train\images\\"+ base_file_name+ ".jpg"
label_file = r"C:\Users\Rowan\Documents\Rowan\Yolo_test\ELDDS1400c5-dataset-3\train\labels\\" + base_file_name +".txt"


# Load YOLO model for cell detection
cell_model = YOLO(CELL_DETECTOR_MODEL)

# Function to draw YOLO labels and cell number on PV
def draw_cell_with_number(pv_img, cell_bbox, cell_idx, label_file=None):
    """
    pv_img: full PV image
    cell_bbox: [x_min, y_min, x_max, y_max] pixel coords of cell
    cell_idx: index of cell (number to display)
    label_file: optional YOLO labels for defects
    """
    x_min, y_min, x_max, y_max = cell_bbox
    cell_w = x_max - x_min
    cell_h = y_max - y_min

    # Draw cell rectangle
    cv2.rectangle(pv_img, (x_min, y_min), (x_max, y_max), (0,255,0), 2)
    cv2.putText(pv_img, f"{cell_idx}", (x_min, y_min+20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)

    # Draw defects if label_file exists
    if label_file and os.path.exists(label_file):
        with open(label_file, "r") as f:
            for line in f:
                parts = line.strip().split()
                class_id = int(parts[0])
                print(class_id)
                x_center, y_center, bw, bh = map(float, parts[1:])
                # Convert normalized cell coords to PV pixel coords
                abs_x1 = int(x_min + (x_center - bw/2) * cell_w)
                abs_y1 = int(y_min + (y_center - bh/2) * cell_h)
                abs_x2 = int(x_min + (x_center + bw/2) * cell_w)
                abs_y2 = int(y_min + (y_center + bh/2) * cell_h)
                cv2.rectangle(pv_img, (abs_x1, abs_y1), (abs_x2, abs_y2), (0,0,255), 2)
                cv2.putText(pv_img, str(class_id), (abs_x1, abs_y1-5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
                
    return pv_img

def draw_cell_with_pv_labels(pv_img, cell_bbox, cell_idx, pv_label_file=None):
    """
    Draws detected cell and any defects from PV-level annotations inside that cell.
    
    pv_img: full PV image
    cell_bbox: [x_min, y_min, x_max, y_max] pixel coords of cell
    cell_idx: number to display on cell
    pv_label_file: YOLO annotations for the full PV (optional)
    """
    x_min, y_min, x_max, y_max = cell_bbox
    cell_w = x_max - x_min
    cell_h = y_max - y_min

    # Draw cell rectangle and index
    cv2.rectangle(pv_img, (x_min, y_min), (x_max, y_max), (0,255,0), 2)
    cv2.putText(pv_img, f"{cell_idx}", (x_min, y_min+20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)

    if pv_label_file and os.path.exists(pv_label_file):
        with open(pv_label_file, "r") as f:
            for line in f:
                parts = line.strip().split()
                class_id = int(parts[0])
                x_center, y_center, bw, bh = map(float, parts[1:])  # normalized PV coords

                # Convert normalized coords to absolute PV pixel coords
                abs_x1 = int((x_center - bw/2) * pv_img.shape[1])
                abs_y1 = int((y_center - bh/2) * pv_img.shape[0])
                abs_x2 = int((x_center + bw/2) * pv_img.shape[1])
                abs_y2 = int((y_center + bh/2) * pv_img.shape[0])

                # Check if the defect is inside the current cell
                tolx = toly = 6
                if abs_x1 >= x_min-tolx and abs_x2 <= x_max+tolx and abs_y1 >= y_min-tolx and abs_y2 <= y_max+tolx:
                    # Draw defect box relative to PV (or optionally relative to cell)
                    cv2.rectangle(pv_img, (abs_x1, abs_y1), (abs_x2, abs_y2), (0,0,255), 2)
                    cv2.putText(pv_img, str(class_id), (abs_x1, abs_y1-5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)

    return pv_img

def draw_annotation(img, cell_idx, label_path):
    h, w = img.shape[:2]

    # Draw YOLO annotations
    with open(label_path, "r") as f:
        for line in f:
            print(cell_idx, line)
            cls, xc, yc, bw, bh = map(float, line.split())

            x1 = int((xc - bw / 2) * w)
            y1 = int((yc - bh / 2) * h)
            x2 = int((xc + bw / 2) * w)
            y2 = int((yc + bh / 2) * h)

            # cls = int(cls)
            # label = class_names[cls]
            # print(label)
            color = (0, 255, 0)
                    
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            # cv2.putText(img, cls, (x1, y1 - 5),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return img

# def temp_draw_annotation():
#     # import yaml
#     # import cv2

#     # Paths
#     data_yaml_path = r'C:\Users\Rowan\Documents\Rowan\Yolo_test\pv-last-version--2-2\data.yaml'
#     label_path = r'C:\Users\Rowan\Documents\Rowan\Yolo_test\pv-last-version--2-2\train\labels\000184_png.rf.9803d36b8edfce5813a899fd7c2fec98.txt'
#     image_path = r'C:\Users\Rowan\Documents\Rowan\Yolo_test\pv-last-version--2-2\train\images\000184_png.rf.9803d36b8edfce5813a899fd7c2fec98.jpg'

#     COLOR_MAP = {
#         "examined": (0, 255, 0),
#         "micro_crack": (0, 255, 255),
#         "shortcircuitcell": (0, 150, 255),
#         "isolated_area": (255, 200, 0),
#         "contamination": (255, 80, 0),
#         "other_error": (255, 150, 0),
#         "crack": (0, 0, 255),
#         "ShortCircuitString": (0, 0, 200),
#         "break": (0, 0, 150)
#     }

#     # Load class names dynamically
#     with open(data_yaml_path, "r") as f:
#         data = yaml.safe_load(f)

#     class_names = data["names"]

#     # Load image
#     img = cv2.imread(image_path)
#     h, w = img.shape[:2]

#     # Draw YOLO annotations
#     with open(label_path, "r") as f:
#         for line in f:
#             cls, xc, yc, bw, bh = map(float, line.split())

#             x1 = int((xc - bw / 2) * w)
#             y1 = int((yc - bh / 2) * h)
#             x2 = int((xc + bw / 2) * w)
#             y2 = int((yc + bh / 2) * h)

#             cls = int(cls)
#             label = class_names[cls]
#             print(label)
#             color = (0, 255, 0)
#             for k in COLOR_MAP.keys():
#                 if k in label.lower():
#                     color = COLOR_MAP[k]
#                 elif 'micro' in label.lower():
#                     color = COLOR_MAP['micro_crack']
                    
#             cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
#             cv2.putText(img, label[:18], (x1, y1 - 5),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # cv2.imwrite(r'output_reports\test.jpg', img)


# Preprocessing function
def preprocess_pv(img, mask_threshold=10):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    inv = cv2.bitwise_not(gray)

    # Detect vertical lines
    vert_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 8))
    thin_vertical = cv2.morphologyEx(inv, cv2.MORPH_OPEN, vert_kernel)

    # Detect wide vertical lines (busbars)
    wide_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 8))
    wide_vertical = cv2.morphologyEx(inv, cv2.MORPH_OPEN, wide_kernel)

    # Only thin vertical lines
    thin_vertical_only = cv2.subtract(thin_vertical, wide_vertical)

    # Threshold mask
    mask = cv2.threshold(thin_vertical_only, mask_threshold, 255, cv2.THRESH_BINARY)[1]

    # Inpaint
    clean = cv2.inpaint(img, mask, 1, cv2.INPAINT_TELEA)

    return clean

# Function to compute mode cell width
def compute_cell_stats(results):
    widths = []
    for det in results[0].boxes.xyxy:
        x_min, y_min, x_max, y_max = map(int, det)
        widths.append(x_max - x_min)
    if not widths:
        return 0, 0  # fallback
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


# def extract_cell_labels_from_pv(
#     pv_label_file,
#     cell_bbox,
#     pv_w,
#     pv_h,
#     tol=0.1
# ):
#     """
#     Returns YOLO labels relative to the cell
#     """
#     x_min, y_min, x_max, y_max = cell_bbox
#     cell_w = x_max - x_min
#     cell_h = y_max - y_min

#     cell_labels = []

#     if not os.path.exists(pv_label_file):
#         return cell_labels

#     with open(pv_label_file, "r") as f:
#         for line in f:
#             cls, xc, yc, bw, bh = map(float, line.strip().split())

#             # PV absolute coords
#             abs_x1 = (xc - bw / 2) * pv_w
#             abs_y1 = (yc - bh / 2) * pv_h
#             abs_x2 = (xc + bw / 2) * pv_w
#             abs_y2 = (yc + bh / 2) * pv_h
#             # Check if defect inside cell
#             if (
#                 abs_x1 >= x_min - tol and
#                 abs_x2 <= x_max + tol and
#                 abs_y1 >= y_min - tol and
#                 abs_y2 <= y_max + tol
#             ):
#                 # Convert to cell-relative coords
#                 cx = ((abs_x1 + abs_x2) / 2 - x_min) / cell_w
#                 cy = ((abs_y1 + abs_y2) / 2 - y_min) / cell_h
#                 cw = (abs_x2 - abs_x1) / cell_w
#                 ch = (abs_y2 - abs_y1) / cell_h

#                 # Clamp safety
#                 cx = min(max(cx, 0), 1)
#                 cy = min(max(cy, 0), 1)
#                 cw = min(max(cw, 0), 1)
#                 ch = min(max(ch, 0), 1)

#                 cell_labels.append(
#                     f"{int(cls)} {cx:.6f} {cy:.6f} {cw:.6f} {ch:.6f}"
#                 )

#     return cell_labels

# def extract_cell_labels_from_pv(pv_label_file, cell_bbox, pv_w, pv_h, min_overlap=0.1):
#     """
#     Returns YOLO labels relative to the cell.
#     Uses overlap instead of strict containment.
#     min_overlap: fraction of PV defect box that must overlap with cell to keep
#     """
#     x_min, y_min, x_max, y_max = cell_bbox
#     cell_w = x_max - x_min
#     cell_h = y_max - y_min
#     cell_area = cell_w * cell_h

#     cell_labels = []

#     if not os.path.exists(pv_label_file):
#         return cell_labels

#     with open(pv_label_file, "r") as f:
#         for line in f:
#             cls, xc, yc, bw, bh = map(float, line.strip().split())

#             # PV absolute coords
#             abs_x1 = (xc - bw / 2) * pv_w
#             abs_y1 = (yc - bh / 2) * pv_h
#             abs_x2 = (xc + bw / 2) * pv_w
#             abs_y2 = (yc + bh / 2) * pv_h

#             # Compute intersection with cell
#             inter_x1 = max(abs_x1, x_min)
#             inter_y1 = max(abs_y1, y_min)
#             inter_x2 = min(abs_x2, x_max)
#             inter_y2 = min(abs_y2, y_max)

#             inter_w = max(0, inter_x2 - inter_x1)
#             inter_h = max(0, inter_y2 - inter_y1)
#             inter_area = inter_w * inter_h

#             defect_area = (abs_x2 - abs_x1) * (abs_y2 - abs_y1)

#             if inter_area / defect_area >= min_overlap:
#                 # Clip the box to the cell
#                 clipped_x1 = inter_x1
#                 clipped_y1 = inter_y1
#                 clipped_x2 = inter_x2
#                 clipped_y2 = inter_y2

#                 # Convert to cell-relative YOLO
#                 cx = ((clipped_x1 + clipped_x2) / 2 - x_min) / cell_w
#                 cy = ((clipped_y1 + clipped_y2) / 2 - y_min) / cell_h
#                 cw = (clipped_x2 - clipped_x1) / cell_w
#                 ch = (clipped_y2 - clipped_y1) / cell_h

#                 # Clamp safety
#                 cx = min(max(cx, 0), 1)
#                 cy = min(max(cy, 0), 1)
#                 cw = min(max(cw, 0), 1)
#                 ch = min(max(ch, 0), 1)

#                 cell_labels.append(
#                     f"{int(cls)} {cx:.6f} {cy:.6f} {cw:.6f} {ch:.6f}"
#                 )

#     return cell_labels

# def extract_cell_labels_from_pv(pv_label_file, filtered_boxes, pv_w, pv_h, max_cells=3, min_overlap=0.3):
#     """
#     Map PV-level defects to cell-level YOLO labels.
    
#     filtered_boxes: list of detected cells [x1, y1, x2, y2]
#     max_cells: ignore defects spanning more than this number of cells
#     min_overlap: minimum fraction of defect area inside a cell to consider
#     """
#     cell_labels_dict = {i: [] for i in range(len(filtered_boxes))}

#     if not os.path.exists(pv_label_file):
#         return cell_labels_dict

#     with open(pv_label_file, "r") as f:
#         for line in f:
#             cls, xc, yc, bw, bh = map(float, line.strip().split())

#             # PV absolute coords
#             abs_x1 = (xc - bw/2) * pv_w
#             abs_y1 = (yc - bh/2) * pv_h
#             abs_x2 = (xc + bw/2) * pv_w
#             abs_y2 = (yc + bh/2) * pv_h

#             defect_area = (abs_x2 - abs_x1) * (abs_y2 - abs_y1)

#             overlaps = []
#             for i, (cx1, cy1, cx2, cy2) in enumerate (filtered_boxes):
#                 inter_x1 = max(abs_x1, cx1)
#                 inter_y1 = max(abs_y1, cy1)
#                 inter_x2 = min(abs_x2, cx2)
#                 inter_y2 = min(abs_y2, cy2)

#                 inter_w = max(0, inter_x2 - inter_x1)
#                 inter_h = max(0, inter_y2 - inter_y1)
#                 inter_area = inter_w * inter_h

#                 if inter_area / defect_area >= min_overlap:
#                     overlaps.append((i, inter_area))

#             # Ignore defects spanning too many cells
#             if 0 < len(overlaps) <= max_cells:
#                 for i, inter_area in overlaps:
#                     cx1, cy1, cx2, cy2 = filtered_boxes[i]
#                     # Clip to cell
#                     clipped_x1 = max(abs_x1, cx1)
#                     clipped_y1 = max(abs_y1, cy1)
#                     clipped_x2 = min(abs_x2, cx2)
#                     clipped_y2 = min(abs_y2, cy2)

#                     cell_w = cx2 - cx1
#                     cell_h = cy2 - cy1

#                     # Convert to cell-relative YOLO
#                     cx_rel = ((clipped_x1 + clipped_x2)/2 - cx1) / cell_w
#                     cy_rel = ((clipped_y1 + clipped_y2)/2 - cy1) / cell_h
#                     cw_rel = (clipped_x2 - clipped_x1) / cell_w
#                     ch_rel = (clipped_y2 - clipped_y1) / cell_h

#                     # Skip tiny boxes
#                     if cw_rel <= 0 or ch_rel <= 0:
#                         continue

#                     cell_labels_dict[i].append(
#                         f"{int(cls)} {cx_rel:.6f} {cy_rel:.6f} {cw_rel:.6f} {ch_rel:.6f}"
#                     )

#     return cell_labels_dict


def extract_cell_labels_from_pv(
    pv_label_file,
    cell_bbox,
    pv_w,
    pv_h,
    tol=6
):
    """
    Returns YOLO labels relative to the cell
    """
    x_min, y_min, x_max, y_max = cell_bbox
    cell_w = x_max - x_min
    cell_h = y_max - y_min

    cell_labels = []

    with open(pv_label_file, "r") as f:
        lines = f.read()

    # Read from the list instead of file (adjusting to your data format)
    lines = lines.strip().split('\n')
    
    for line in lines:
        parts = line.strip().split()
        print(parts)
        cls = int(float(parts[0]))
        xc, yc, bw, bh = map(float, parts[1:5])

        # PV absolute coords (from normalized coordinates)
        # The annotations are already normalized, so convert to pixel coordinates
        abs_x1 = (xc - bw / 2) * pv_w
        abs_y1 = (yc - bh / 2) * pv_h
        abs_x2 = (xc + bw / 2) * pv_w
        abs_y2 = (yc + bh / 2) * pv_h
        
        # Convert tolerance from pixels to normalized units
        # (since your cell_bbox is likely in pixels)
        tol_norm_x = tol / pv_w
        tol_norm_y = tol / pv_h
        
        # Convert cell_bbox to normalized coordinates for comparison
        x_min_norm = x_min / pv_w
        y_min_norm = y_min / pv_h
        x_max_norm = x_max / pv_w
        y_max_norm = y_max / pv_h
        
        # Check if defect center is inside cell (more reliable than checking entire box)
        # or if there's significant overlap
        center_x_norm = xc
        center_y_norm = yc
        
        # Method 1: Check if center is inside cell
        if (x_min_norm <= center_x_norm <= x_max_norm and 
            y_min_norm <= center_y_norm <= y_max_norm):
            
            # Convert to cell-relative coords
            # First, clip the bounding box to cell boundaries
            clip_x1 = max(abs_x1, x_min)
            clip_y1 = max(abs_y1, y_min)
            clip_x2 = min(abs_x2, x_max)
            clip_y2 = min(abs_y2, y_max)
            
            # Calculate cell-relative coordinates
            if clip_x2 > clip_x1 and clip_y2 > clip_y1:  # Valid intersection
                cx = ((clip_x1 + clip_x2) / 2 - x_min) / cell_w
                cy = ((clip_y1 + clip_y2) / 2 - y_min) / cell_h
                cw = (clip_x2 - clip_x1) / cell_w
                ch = (clip_y2 - clip_y1) / cell_h
                
                # Only keep if the annotation is reasonably sized
                if cw > 0.01 and ch > 0.01:  # At least 1% of cell size
                    # Clamp safety
                    cx = min(max(cx, 0), 1)
                    cy = min(max(cy, 0), 1)
                    cw = min(max(cw, 0), 1)
                    ch = min(max(ch, 0), 1)
                    
                    cell_labels.append(
                        f"{int(cls)} {cx:.6f} {cy:.6f} {cw:.6f} {ch:.6f}"
                    )

    return cell_labels


def extract_cell_labels_from_pv(
    pv_label_file,
    cell_bbox,
    pv_w,
    pv_h,
    tol=20
):
    """
    Returns YOLO labels relative to the cell
    Args:
        pv_label_file: Path to PV annotation file (normalized coords)
        cell_bbox: [x_min, y_min, x_max, y_max] in pixels (from YOLO xyxy)
        pv_w: Width of PV image in pixels
        pv_h: Height of PV image in pixels
        tol: Tolerance in pixels for boundary checking
    """
    x_min, y_min, x_max, y_max = cell_bbox
    cell_w = x_max - x_min
    cell_h = y_max - y_min

    cell_labels = []

    # Read PV annotations (normalized coordinates)
    with open(pv_label_file, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
                
            cls = int(float(parts[0]))
            xc, yc, bw, bh = map(float, parts[1:5])

            # Convert normalized PV annotations to absolute pixel coordinates
            abs_x1 = (xc - bw / 2) * pv_w
            abs_y1 = (yc - bh / 2) * pv_h
            abs_x2 = (xc + bw / 2) * pv_w
            abs_y2 = (yc + bh / 2) * pv_h
            
            # Calculate center of PV annotation
            center_x = (abs_x1 + abs_x2) / 2
            center_y = (abs_y1 + abs_y2) / 2
            
            # Check if center is inside cell (with tolerance)
            if (x_min - tol <= center_x <= x_max + tol and 
                y_min - tol <= center_y <= y_max + tol):
                
                # Convert to cell-relative coordinates
                # Clip bounding box to cell boundaries
                clip_x1 = max(abs_x1, x_min)
                clip_y1 = max(abs_y1, y_min)
                clip_x2 = min(abs_x2, x_max)
                clip_y2 = min(abs_y2, y_max)
                
                # Calculate relative coordinates
                cx = ((clip_x1 + clip_x2) / 2 - x_min) / cell_w
                cy = ((clip_y1 + clip_y2) / 2 - y_min) / cell_h
                cw = (clip_x2 - clip_x1) / cell_w
                ch = (clip_y2 - clip_y1) / cell_h
                
                # Only add if annotation has reasonable size
                if cw > 0.01 and ch > 0.01:
                    # Clamp to [0, 1]
                    cx = max(0, min(1, cx))
                    cy = max(0, min(1, cy))
                    cw = max(0, min(1, cw))
                    ch = max(0, min(1, ch))
                    
                    cell_labels.append(
                        f"{cls} {cx:.6f} {cy:.6f} {cw:.6f} {ch:.6f}"
                    )

    return cell_labels

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

    
# cv2.imshow("PV with cell numbers", pv_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


train_img_dir = r"ELDDS1400c5-dataset-3/train/images"
train_lbl_dir = r"ELDDS1400c5-dataset-3/train/labels"

out_img_dir = r"cells_ELDDS1400c5-dataset-3/train/images"
out_lbl_dir = r"cells_ELDDS1400c5-dataset-3/train/labels"

os.makedirs(out_img_dir, exist_ok=True)
os.makedirs(out_lbl_dir, exist_ok=True)

image_files = sorted(os.listdir(train_img_dir))[:5]  # 🔥 test on 5 only

for img_name in image_files:
    img_path = os.path.join(train_img_dir, img_name)
    lbl_path = os.path.join(
        train_lbl_dir,
        os.path.splitext(img_name)[0] + ".txt"
    )

    pv_img = cv2.imread(img_path)
    pv_h, pv_w = pv_img.shape[:2] 
    resized_img = cv2.resize(pv_img, (IMAGE_SIZE, IMAGE_SIZE))

    clean = preprocess_pv(resized_img, mask_threshold=10)

    # Run cell detection
    results = cell_model.predict(clean, conf=0.4, iou=0.6, imgsz=IMAGE_SIZE)

    # Compute cell stats
    mode_width, all_widths = compute_cell_stats(results)
    print(mode_width)
    # print(all_widths)

    # Define abnormal thresholds
    too_wide_thresh = mode_width * 1.5
    too_narrow_thresh = mode_width * 0.5

    # Count abnormal cells
    num_wide = sum(1 for w in all_widths if w > too_wide_thresh)
    num_narrow = sum(1 for w in all_widths if w < too_narrow_thresh)

    # Decision logic
    if num_wide > len(all_widths) * 0.2:  # >20% cells are too wide
        print("Too many wide cells → rerunning preprocessing with lower threshold")
        clean = preprocess_pv(resized_img, mask_threshold=8)
        # Run cell detection
        results = cell_model.predict(clean, conf=0.4, iou=0.6,  imgsz=IMAGE_SIZE)

        # Compute cell stats
        mode_width, all_widths = compute_cell_stats(results)
        print(mode_width)
        print(all_widths)

    elif num_narrow > len(all_widths) * 0.2:  # >20% too narrow
        print("Many small cells → pass as is")
    else:
        print("Cell sizes OK → proceed")

    # --- Usage with YOLO results ---
    boxes = results[0].boxes.xyxy.cpu().numpy()  # list of detected boxes
    boxes = [list(b) for b in boxes]  # convert to Python list if needed

    # Remove outliers (±30% from mode)
    # print(len(boxes))
    filtered_boxes = filter_boxes_by_mode(boxes)

    sorted_boxes = sort_boxes_top_down_left_right(filtered_boxes)

    # print(len(filtered_boxes))

    # # Now use filtered_boxes for visualization or further processing
    # for i, det in enumerate(filtered_boxes):
    #     x_min, y_min, x_max, y_max = map(int, det)
    #     pv_img = draw_cell_with_pv_labels(clean, [x_min, y_min, x_max, y_max], i, label_file)
        
    # pv_img = draw_cell_with_pv_labels(clean, filtered_boxes,  label_file)
    # filtered_boxes = your outlier-removed YOLO boxes

    # cell_labels_dict = extract_cell_labels_from_pv(lbl_path, filtered_boxes, pv_w, pv_h)

    # # Then save each cell crop + corresponding labels
    # for i, det in enumerate(filtered_boxes):
    #     x1, y1, x2, y2 = map(int, det)
    #     cell_crop = resized_img[y1:y2, x1:x2]
    #     if cell_crop.size == 0:
    #         continue

    #     # Draw on PV for visualization
    #     cv2.rectangle(resized_img, (x1, y1), (x2, y2), (0,255,255), 2)
    #     cv2.putText(resized_img, str(i), (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,255), 1)

    #     # Save image
    #     cell_img_name = f"{os.path.splitext(img_name)[0]}_cell{i}.jpg"
    #     cv2.imwrite(os.path.join(out_img_dir, cell_img_name), cell_crop)

    #     # Save labels from the dict
    #     cell_lbl_name = f"{os.path.splitext(img_name)[0]}_cell{i}.txt"
    #     with open(os.path.join(out_lbl_dir, cell_lbl_name), "w") as f:
    #         f.write("\n".join(cell_labels_dict.get(i, [])))
    #     cell_lbl_file = os.path.join(out_lbl_dir, cell_lbl_name)

    #     draw_annotation(cell_crop, i, cell_lbl_file)
            

    for i, det in enumerate(filtered_boxes):
        x1, y1, x2, y2 = map(int, det)
        cell_crop = resized_img[y1:y2, x1:x2]
        if cell_crop.size == 0:
            continue
        
        cv2.rectangle(resized_img, (x1, y1), (x2, y2), (0,255,255), 2)
        cv2.putText(
        resized_img,
        str(i),
        (x1, y1-5),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        (0,255,255),
        1
    )
        # cv2.imshow("PV with cell numbers", resized_img)
        # cv2.waitKey(0)

        cell_img_name = f"{os.path.splitext(img_name)[0]}_cell{i}.jpg"
        cell_lbl_name = f"{os.path.splitext(img_name)[0]}_cell{i}.txt"

        cv2.imwrite(os.path.join(out_img_dir, cell_img_name), cell_crop)

        cell_labels = extract_cell_labels_from_pv(
            lbl_path,
            det,
            pv_w,
            pv_h
        )
        
        cell_lbl_file = os.path.join(out_lbl_dir, cell_lbl_name)

        if len(cell_labels) > 0:
            print(cell_img_name)
            with open(cell_lbl_file, "w") as f:
                f.write("\n".join(cell_labels))


            cv2.imshow("Cell annotation", draw_annotation(cell_crop, i, cell_lbl_file))
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    cv2.imshow("PV with cell numbers", resized_img)
    cv2.waitKey(0)
    print(f"Processed {img_name}")