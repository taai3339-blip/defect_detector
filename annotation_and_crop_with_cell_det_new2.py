import cv2
import os
from ultralytics import YOLO
import numpy as np
from collections import Counter

# Paths
# pv_images_path = r"modified_data_1/valid/images"
# pv_labels_path = r"modified_data_1/valid/labels"  # YOLO-format PV-level annotations
# pv_images_path = r"modified_data_1/valid/images"
# pv_labels_path = r"modified_data_1/valid/labels"  # YOLO-format PV-level annotations

# CELL_DETECTOR_MODEL = r"C:\Users\Rowan\Documents\Final_demo\Final_demo\demo\demo\demo\demo\src\main\backend\models/best2.pt"  # YOLO trained to detect cells
# # CELL_DETECTOR_MODEL = r"runs\detect\train9\weights\best.pt"  # YOLO trained to detect cells
# # CELL_DETECTOR_MODEL = r"models\cell_detector_imgs1024.ptt"
# CELL_DETECTOR_MODEL = r"runs\detect\train3\weights\best.pt"

# IMAGE_SIZE = 960
# CELL_PADDING = 4  # Number of pixels to add around each detected cell for context.
#                   # Annotations remain relative to the original cell dimensions (without padding).
#                   # Set to 0 to disable padding.


# output_defect_path = "output_cells_defect"
# output_clean_path = "output_cells_clean"
# base_file_name = "B085_png.rf.ee886bfe6746351a81363739a658b120"
# base_file_name = "C209AABB3A20190033_png.rf.24e6c7a051a024707e10d13b194e5c11"

# # Example: visualize one PV image with cell numbers
# img_file = os.listdir(pv_images_path)[0]  # first PV image
# pv_img_path = os.path.join(pv_images_path, img_file)

# pv_img_path = r"C:\Users\Rowan\Documents\Rowan\Yolo_test\ELDDS1400c5-dataset-3\train\images\\"+ base_file_name+ ".jpg"
# label_file = r"C:\Users\Rowan\Documents\Rowan\Yolo_test\ELDDS1400c5-dataset-3\train\labels\\" + base_file_name +".txt"

# train_img_dir = r"ELDDS1400c5-dataset-3/test/images"
# train_lbl_dir = r"ELDDS1400c5-dataset-3/test/labels"

# out_img_dir = r"cells_ELDDS1400c5-dataset-3/test/images"
# out_lbl_dir = r"cells_ELDDS1400c5-dataset-3/test/labels"

# os.makedirs(out_img_dir, exist_ok=True)
# os.makedirs(out_lbl_dir, exist_ok=True)

# # Load YOLO model for cell detection
# cell_model = YOLO(CELL_DETECTOR_MODEL)


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
    cv2.rectangle(pv_img, (x_min, y_min), (x_max, y_max), (255,0,0), 2)
    # cv2.putText(pv_img, f"{cell_idx}", (x_min, y_min+20),
    #             cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)

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
                    
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 1)
            # cv2.putText(img, cls, (x1, y1 - 5),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return img



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


def extract_cell_labels_from_pv(
    pv_label_file,
    cell_bbox,
    pv_w,
    pv_h,
    resized_w,
    resized_h,
    pad_x=0,
    pad_y=0,
    cell_w_orig=None,
    cell_h_orig=None,
    crop_w=None,
    crop_h=None,
    min_overlap_ratio=0.1
):
    """
    Returns YOLO labels relative to the cell crop (padded if padding is used)
    Args:
        pv_label_file: Path to PV annotation file (normalized coords relative to original image)
        cell_bbox: [x_min, y_min, x_max, y_max] in pixels (from YOLO xyxy, relative to resized image)
        pv_w: Width of original PV image in pixels
        pv_h: Height of original PV image in pixels
        resized_w: Width of resized image used for cell detection
        resized_h: Height of resized image used for cell detection
        pad_x: X padding offset in the crop (position of original cell x1 in the crop)
        pad_y: Y padding offset in the crop (position of original cell y1 in the crop)
        cell_w_orig: Original cell width without padding (in resized image, None if no padding)
        cell_h_orig: Original cell height without padding (in resized image, None if no padding)
        crop_w: Width of the padded crop (in resized image, None if no padding)
        crop_h: Height of the padded crop (in resized image, None if no padding)
        min_overlap_ratio: Minimum fraction of defect area that must overlap with cell (0.0 to 1.0)
    """
    # Scale cell bbox from resized image coordinates to original image coordinates
    scale_x = pv_w / resized_w
    scale_y = pv_h / resized_h
    
    x_min_resized, y_min_resized, x_max_resized, y_max_resized = cell_bbox
    x_min = x_min_resized * scale_x
    y_min = y_min_resized * scale_y
    x_max = x_max_resized * scale_x
    y_max = y_max_resized * scale_y
    
    cell_w = x_max - x_min
    cell_h = y_max - y_min
    cell_area = cell_w * cell_h
    
    # Determine if we're using padding
    use_padding = (cell_w_orig is not None and cell_h_orig is not None and 
                   crop_w is not None and crop_h is not None)

    cell_labels = []

    if not os.path.exists(pv_label_file):
        return cell_labels

    # Read PV annotations (normalized coordinates relative to original image)
    with open(pv_label_file, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
                
            cls = int(float(parts[0]))
            xc, yc, bw, bh = map(float, parts[1:5])

            # Convert normalized PV annotations to absolute pixel coordinates (original image size)
            defect_x1 = (xc - bw / 2) * pv_w
            defect_y1 = (yc - bh / 2) * pv_h
            defect_x2 = (xc + bw / 2) * pv_w
            defect_y2 = (yc + bh / 2) * pv_h
            
            defect_w = defect_x2 - defect_x1
            defect_h = defect_y2 - defect_y1
            defect_area = defect_w * defect_h
            
            # Calculate intersection between defect box and cell box (original cell, not padded)
            inter_x1 = max(defect_x1, x_min)
            inter_y1 = max(defect_y1, y_min)
            inter_x2 = min(defect_x2, x_max)
            inter_y2 = min(defect_y2, y_max)
            
            inter_w = max(0, inter_x2 - inter_x1)
            inter_h = max(0, inter_y2 - inter_y1)
            inter_area = inter_w * inter_h
            
            # Check if there's sufficient overlap
            if defect_area > 0 and inter_area / defect_area >= min_overlap_ratio:
                # Clip the defect box to cell boundaries
                clip_x1 = inter_x1
                clip_y1 = inter_y1
                clip_x2 = inter_x2
                clip_y2 = inter_y2
                
                if use_padding:
                    # Get position relative to original cell (0-1 normalized to original cell dimensions)
                    cx_rel_cell = ((clip_x1 + clip_x2) / 2 - x_min) / cell_w
                    cy_rel_cell = ((clip_y1 + clip_y2) / 2 - y_min) / cell_h
                    cw_rel_cell = (clip_x2 - clip_x1) / cell_w
                    ch_rel_cell = (clip_y2 - clip_y1) / cell_h
                    
                    # Convert to position in the padded crop (in resized image coordinates)
                    # In the resized image, original cell is at (x_min_resized, y_min_resized)
                    # In the crop, original cell is at (pad_x, pad_y) with size (cell_w_orig, cell_h_orig)
                    # Scale the relative coordinates to the resized image
                    cell_w_resized = x_max_resized - x_min_resized
                    cell_h_resized = y_max_resized - y_min_resized
                    
                    # Position relative to original cell in resized image
                    x_in_cell_resized = cx_rel_cell * cell_w_resized
                    y_in_cell_resized = cy_rel_cell * cell_h_resized
                    w_in_cell_resized = cw_rel_cell * cell_w_resized
                    h_in_cell_resized = ch_rel_cell * cell_h_resized
                    
                    # Position in the crop (crop starts at crop_x1, crop_y1 in resized image)
                    x_in_crop = pad_x + x_in_cell_resized
                    y_in_crop = pad_y + y_in_cell_resized
                    
                    # Normalize relative to crop dimensions (in resized image coordinates)
                    cx = x_in_crop / crop_w
                    cy = y_in_crop / crop_h
                    cw = w_in_cell_resized / crop_w
                    ch = h_in_cell_resized / crop_h
                else:
                    # No padding: annotations relative to original cell dimensions
                    cx = ((clip_x1 + clip_x2) / 2 - x_min) / cell_w
                    cy = ((clip_y1 + clip_y2) / 2 - y_min) / cell_h
                    cw = (clip_x2 - clip_x1) / cell_w
                    ch = (clip_y2 - clip_y1) / cell_h
                
                # Only add if annotation has reasonable size (at least 1% of cell in each dimension)
                if cw > 0.01 and ch > 0.01:
                    # Clamp to [0, 1] to ensure valid YOLO format
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



# --- 1. HELPER FUNCTIONS ---

def order_points(pts):
    """
    Sorts the 4 polygon points into: Top-Left, Top-Right, Bottom-Right, Bottom-Left
    This is required for Perspective Transform.
    """
    rect = np.zeros((4, 2), dtype="float32")

    # Sum of coordinates (x+y)
    # Smallest sum = Top-Left
    # Largest sum = Bottom-Right
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # Difference of coordinates (y-x)
    # Smallest diff = Top-Right
    # Largest diff = Bottom-Left
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect

def add_margin(poly, margin_ratio=0.05):
    """
    Expands the polygon outward by a specific ratio (e.g., 5%).
    """
    # Calculate centroid
    centroid = np.mean(poly, axis=0)
    
    # Scale points away from centroid
    # Formula: NewPoint = Centroid + (OldPoint - Centroid) * (1 + Margin)
    scaled_poly = centroid + (poly - centroid) * (1 + margin_ratio)
    
    return scaled_poly.astype(np.float32)

def four_point_transform(image, pts):
    """
    Crops and warps the image based on 4 points.
    """
    if np.isnan(pts).any() or np.isinf(pts).any():
        return None
    
    if pts.shape != (4,2):
        return None

    # 1. Order the points
    rect = order_points(pts)

    # 2. Calculate new width and height of the image
    widthA = np.sqrt(((rect[2][0] - rect[3][0]) ** 2) + ((rect[2][1] - rect[3][1]) ** 2))
    widthB = np.sqrt(((rect[1][0] - rect[0][0]) ** 2) + ((rect[1][1] - rect[0][1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((rect[1][0] - rect[2][0]) ** 2) + ((rect[1][1] - rect[2][1]) ** 2))
    heightB = np.sqrt(((rect[0][0] - rect[3][0]) ** 2) + ((rect[0][1] - rect[3][1]) ** 2))
 
    maxHeight = max(int(heightA), int(heightB))
    if maxWidth == 0 or maxHeight == 0:
        return None
    # 3. Destination points (Top-Left, Top-Right, Bottom-Right, Bottom-Left)
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    # 4. Calculate the perspective transform matrix and warp
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    return warped

# --- 2. THE MAIN PROCESSING FUNCTION ---

def process_pv_images(input_folder, output_cropped_folder, output_not_found_folder):
    """
    Goes through all images in input_folder.
    - If PV found: Adds margin, crops straight, saves to output_cropped_folder.
    - If PV not found: Saves original to output_not_found_folder.
    """
    
    # Create output directories if they don't exist
    os.makedirs(output_cropped_folder, exist_ok=True)
    os.makedirs(output_not_found_folder, exist_ok=True)

    # Supported image extensions
    valid_exts = ('.jpg', '.jpeg', '.png', '.bmp')
    
    image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(valid_exts)]
    print(f"Found {len(image_files)} images to process.")

    for filename in image_files:
        filepath = os.path.join(input_folder, filename)
        img = cv2.imread(filepath)

        if img is None:
            continue

        # --- DETECTION STEP ---
        # Using the "Loose" function we built previously
        poly = get_module_polygon(img)

        # --- HANDLING RESULTS ---
        if poly is not None:
            # 1. Add Margin (e.g., 5% larger)
            margin_ratio = 0.05 
            expanded_poly = add_margin(poly, margin_ratio)

            # 2. Crop and Straighten
            cropped_img = four_point_transform(img, expanded_poly)

            # 3. Save to "Cropped" folder
            save_path = os.path.join(output_cropped_folder, filename)
            cv2.imwrite(save_path, cropped_img)
            print(f"[OK] Processed: {filename}")
            
        else:
            # 1. No polygon found -> Save Original to "Not Found" folder
            save_path = os.path.join(output_not_found_folder, filename)
            cv2.imwrite(save_path, img)
            print(f"[MISSING] Saved original: {filename}")

# --- 3. EXECUTION ---
# Make sure 'get_module_polygon_loose' is defined in your script (from previous answer)



def get_module_polygon(image, target_height=640):
    """
    Detects the 4 corners of the PV module frame in an EL image.
    Returns: A numpy array of 4 points (x, y) defining the polygon, or None if failed.
    """
    # 1. Preprocessing
    original_h , original_w = image.shape[:2]

    scale_factor = target_height / original_h

    small_image = cv2.resize(image, (0, 0), fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_AREA)

    # Convert to grayscale if needed
    if len(small_image.shape) == 3:
        gray = cv2.cvtColor(small_image, cv2.COLOR_BGR2GRAY)
    else:
        gray = small_image
    
    if np.mean(gray) < 50:
        gray = cv2.bitwise_not(gray)
    # Apply Gaussian Blur to reduce "snowy" noise common in EL images
    blurred = cv2.GaussianBlur(gray, (15, 15), 0)

    # 2. Thresholding (Separate Module from Background)
    # Otsu's method automatically finds the best threshold value.
    # We use THRESH_BINARY_INV because the background is usually dark (low value) 
    # and the module is bright. This makes the background BLACK and module WHITE.
    _, thresh_otsu = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    white_ratio=np.sum(thresh_otsu==255) / thresh_otsu.size
    if white_ratio > 0.95 or white_ratio <0.05:
        _, thresh = cv2.threshold(blurred, 100, 255, cv2.THRESH_BINARY)
    else:
        thresh = thresh_otsu
    # 3. Morphological Operations (Fill gaps)
    # EL frames can sometimes be faint or broken. 
    # Closing (Dilation followed by Erosion) merges nearby white regions.
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

    # 4. Find Contours
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None # No module found

    # Assume the module is the LARGEST contour by area
    # This filters out small noise specs or hotspots
    largest_contour = max(contours, key=cv2.contourArea)

    small_h, small_w = small_image.shape[:2]
    
    best_cnt = find_best_pv_contour(contours, small_w, small_h)

    if best_cnt is None:
        return None
    
    

    # 5. Approximate to a Polygon (Get the 4 corners)
    # This handles rotation and perspective distortion
    perimeter = cv2.arcLength(best_cnt, True)
    
    # 0.02 is an epsilon value. Adjust it (0.01 to 0.05) if it detects too few or too many corners.
    approx = cv2.approxPolyDP(best_cnt, 0.02 * perimeter, True)

    # Ensure we have 4 corners
    # If lighting is terrible, we might get more points. Fallback to convex hull if needed.
    if 4<= len(approx) <=8 :
        hull = cv2.convexHull(approx)
        peri=cv2.arcLength(hull, True)
        final_approx = cv2.approxPolyDP(hull, 0.02*peri, True)
        if len(final_approx) ==4:

            scaled_poly = (final_approx.reshape(-1,2)/ scale_factor).astype(int)

            return  scaled_poly #approx.reshape(4, 2)
    else:
        # Fallback: If approximation failed to find 4 corners (e.g. wavy frame), 
        # use the bounding rectangle, though it won't handle perspective as well.
        rect = cv2.minAreaRect(best_cnt)
        box = cv2.boxPoints(rect)
        if np.isnan(box).any() or np.isinf(box).any():
            return None
        scaled_box = (box / scale_factor).astype(int)

        return scaled_box #np.int32(box)

# --- USAGE EXAMPLE: Filtering YOLO Detections ---

def filter_detections_inside_module(image, yolo_boxes):
    """
    image: The input image (numpy array)
    yolo_boxes: List of YOLO detections [ [x1, y1, x2, y2], ... ]
    Returns: List of boxes that are strictly inside the PV module.
    """
    
    # 1. Get the Module Polygon
    module_poly = get_module_polygon(image)
    
    if module_poly is None:
        print("Module frame not found. Returning all boxes (or none based on your logic).")
        return yolo_boxes # Or return [] if you want to be safe

    # 2. Filter Boxes
    valid_boxes = []
    
    # Create a mask for visualization (optional)
    debug_img = image.copy()
    cv2.drawContours(debug_img, [module_poly], -1, (0, 255, 0), 3)

    for box in yolo_boxes:
        # Calculate center of the detection
        # box format: [x_min, y_min, x_max, y_max] (adjust based on your YOLO output)
        center_x = int((box[0] + box[2]) / 2)
        center_y = int((box[1] + box[3]) / 2)
        
        # 3. Check if point is inside polygon
        # pointPolygonTest returns +ve if inside, -ve if outside, 0 if on edge
        distance = cv2.pointPolygonTest(module_poly, (center_x, center_y), measureDist=False)
        
        if distance >= 0:
            valid_boxes.append(box)
        else:
            # Optional: Draw rejected boxes in red
            # cv2.rectangle(debug_img, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 2)
            pass

    # Show the result (Debugging)
    # cv2.imshow("Module Detection", debug_img)
    # cv2.waitKey(0)
    
    return valid_boxes

# --- TEST WITH A DUMMY IMAGE ---
# To test this, load one of your images:

import cv2
import numpy as np

def crop_and_warp_cell(image, box_coords):
    """
    image: The full EL image
    box_coords: [x1, y1, x2, y2] from YOLO model
    """
    x1, y1, x2, y2 = map(int, box_coords)
    
    # 1. Crop the cell based on YOLO box (Add a small margin to be safe)
    margin = 5
    # Ensure coordinates are within image bounds
    h, w = image.shape[:2]
    x1 = max(0, x1 - margin)
    y1 = max(0, y1 - margin)
    x2 = min(w, x2 + margin)
    y2 = min(h, y2 + margin)
    
    cropped_region = image[y1:y2, x1:x2]
    
    # If crop is empty or too small, return None
    if cropped_region.size == 0:
        return None

    # 2. Pre-processing for Edge Detection (Crucial for EL images)
    gray = cv2.cvtColor(cropped_region, cv2.COLOR_BGR2GRAY)
    
    # Apply Threshold to isolate the bright cell from dark background
    # You might need to tune 50 (threshold) depending on image brightness
    _, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)
    
    # Optional: Blur to reduce noise before finding contours
    thresh = cv2.GaussianBlur(thresh, (5, 5), 0)

    # 3. Find Contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None # Fallback or handle error

    # Get the largest contour (should be the cell)
    c = max(contours, key=cv2.contourArea)
    
    # 4. Approximate the contour to a quadrilateral (get the 4 corners)
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02 * peri, True)
    
    # If we didn't find 4 corners, fallback to using the bounding box rect
    if len(approx) != 4:
        # print("Could not find 4 corners, using bounding box")
        return cropped_region # Returns the tilted crop (not ideal but safe)

    # 5. Sort the corners (Top-Left, Top-Right, Bottom-Right, Bottom-Left)
    # Reshape points to (4, 2)
    pts = approx.reshape(4, 2)
    
    # Sort points logic
    rect = np.zeros((4, 2), dtype="float32")
    
    # Sum of coordinates (smallest = Top-Left, largest = Bottom-Right)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    
    # Difference of coordinates (smallest = Top-Right, largest = Bottom-Left)
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    
    # 6. Calculate the new width and height of the "un-tilted" image
    widthA = np.sqrt(((rect[2][0] - rect[3][0]) ** 2) + ((rect[2][1] - rect[3][1]) ** 2))
    widthB = np.sqrt(((rect[1][0] - rect[0][0]) ** 2) + ((rect[1][1] - rect[0][1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    
    heightA = np.sqrt(((rect[1][0] - rect[2][0]) ** 2) + ((rect[1][1] - rect[2][1]) ** 2))
    heightB = np.sqrt(((rect[0][0] - rect[3][0]) ** 2) + ((rect[0][1] - rect[3][1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    
    # 7. Destination points for the warp
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")
    
    # 8. Perform the Perspective Warp
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(cropped_region, M, (maxWidth, maxHeight))
    
    return warped


# ----- Main workflow -----
def main():
    image_files = sorted(os.listdir(train_img_dir)) # [:5]  test on 5 only

    for img_name in image_files:
        img_path = os.path.join(train_img_dir, img_name)
        lbl_path = os.path.join(
            train_lbl_dir,
            os.path.splitext(img_name)[0] + ".txt"
        )

        pv_img = cv2.imread(img_path)
        pv_h, pv_w = pv_img.shape[:2] 
        # resized_img = cv2.resize(pv_img, (IMAGE_SIZE, IMAGE_SIZE))

        # clean = preprocess_pv(resized_img, mask_threshold=10)

        # Run cell detection
        results = cell_model.predict(pv_img, conf=0.4, iou=0.6, imgsz=IMAGE_SIZE)

        # Compute cell stats
        mode_width, all_widths = compute_cell_stats(results)
        # print(mode_width)
        # print(all_widths)

        if len(all_widths) == 0:
            print("no widths. continue")
            continue

        # Define abnormal thresholds
        too_wide_thresh = mode_width * 1.5
        too_narrow_thresh = mode_width * 0.5

        # Count abnormal cells
        num_wide = sum(1 for w in all_widths if w > too_wide_thresh)
        num_narrow = sum(1 for w in all_widths if w < too_narrow_thresh)

        # Decision logic
        if num_wide > len(all_widths) * 0.2:  # >20% cells are too wide
            print("Too many wide cells → rerunning preprocessing with lower threshold")
            clean = preprocess_pv(pv_img, mask_threshold=8)
            # Run cell detection
            results = cell_model.predict(clean, conf=0.4, iou=0.6,  imgsz=IMAGE_SIZE)

            # Compute cell stats
            mode_width, all_widths = compute_cell_stats(results)
            # print(mode_width)
            # print(all_widths)

        elif num_narrow > len(all_widths) * 0.2:  # >20% too narrow
            print("Many small cells → pass as is")
        # else:
        #     print("Cell sizes OK → proceed")

        # --- Usage with YOLO results ---
        boxes = results[0].boxes.xyxy.cpu().numpy()  # list of detected boxes
        boxes = [list(b) for b in boxes]  # convert to Python list if needed

        # print(len(boxes))
        filtered_boxes = filter_boxes_by_mode(boxes)        

        sorted_boxes = sort_boxes_top_down_left_right(filtered_boxes)


        for i, det in enumerate(filtered_boxes):
            x1, y1, x2, y2 = map(int, det)
            
            # Add padding around the cell for context
            pad = CELL_PADDING
            h_resized, w_resized = pv_img.shape[:2]
            
            # Calculate padded coordinates, clamping to image boundaries
            x1_padded = max(0, x1 - pad)
            y1_padded = max(0, y1 - pad)
            x2_padded = min(w_resized, x2 + pad)
            y2_padded = min(h_resized, y2 + pad)
            
            # Crop with padding
            cell_crop = pv_img[y1_padded:y2_padded, x1_padded:x2_padded]
            if cell_crop.size == 0:
                continue
            
            # Original cell dimensions (without padding) in the padded crop
            cell_w_orig = x2 - x1
            cell_h_orig = y2 - y1
            crop_w = x2_padded - x1_padded
            crop_h = y2_padded - y1_padded
            
            # Calculate padding offset in the actual crop (may be less than pad if near image edge)
            pad_x = x1 - x1_padded
            pad_y = y1 - y1_padded
            

            cell_img_name = f"{os.path.splitext(img_name)[0]}_cell{i}.jpg"
            cell_lbl_name = f"{os.path.splitext(img_name)[0]}_cell{i}.txt"

            cv2.imwrite(os.path.join(out_img_dir, cell_img_name), cell_crop)

            cell_labels = extract_cell_labels_from_pv(
                lbl_path,
                det,
                pv_w,
                pv_h,
                IMAGE_SIZE,  # resized width
                IMAGE_SIZE,  # resized height
                pad_x,       # x padding offset in the crop (x1 - x1_padded)
                pad_y,       # y padding offset in the crop (y1 - y1_padded)
                cell_w_orig, # original cell width (without padding, in resized image)
                cell_h_orig, # original cell height (without padding, in resized image)
                crop_w,      # padded crop width (in resized image)
                crop_h       # padded crop height (in resized image)
            )
            
            cell_lbl_file = os.path.join(out_lbl_dir, cell_lbl_name)

            if len(cell_labels) > 0:
                with open(cell_lbl_file, "w") as f:
                    f.write("\n".join(cell_labels))






# Example usage:
input_dir = r'C:\Users\Rowan\Documents\Rowan\all clean'
out_cropped = "cropped_clean_pvs"
out_failed = r"cropped_clean_pvs\not_found_pvs"

# process_pv_images(input_dir, out_cropped, out_failed)

# exit()

# train_img_dir = r'C:\Users\Rowan\Documents\Rowan\Yolo_test\test-anomaly'
# # train_img_dir = r'C:\Users\Rowan\Documents\Rowan\Yolo_test\cropped_clean_pvs'
# out_img_dir = r'test_anomaly_cells_aug_with_rect'
# # out_img_dir = r'cell_detect_level_clean_for_test_aug' # bad
# out_img_dir = r'test_anomaly_pvs'

train_img_dir = r"C:\Users\Rowan\Documents\Rowan\Yolo_test\modified_mixed_pv_orig_res\train\images"
train_lbl_dir = r"C:\Users\Rowan\Documents\Rowan\Yolo_test\modified_mixed_pv_orig_res\train\labels"
out_img_dir = r'C:\Users\Rowan\Documents\Rowan\Yolo_test\cropped_mixed_orig_res_for_defect_yolotrain54\train\images'
out_lbl_dir = r'C:\Users\Rowan\Documents\Rowan\Yolo_test\cropped_mixed_orig_res_for_defect_yolotrain54\train\labels'

# train_img_dir = r'C:\Users\Rowan\Documents\Rowan\all clean'
# out_img_dir = r'cropped_clean_for_anomaly_padding_inner'

IMAGE_SIZE = 960
CELL_MODEL_PATH = r"models\cell_detector_best_model_8s_7_1.pt"

# cell_model = YOLO(CELL_DETECTOR_MODEL)
cell_model = YOLO(CELL_MODEL_PATH)

os.makedirs(out_img_dir, exist_ok=True)
os.makedirs(out_lbl_dir, exist_ok=True)

main()
exit()

image_files = sorted(os.listdir(train_img_dir)) # [:5]  test on 5 only

CELL_PADDING = 4

# CELL_DETECTOR_MODEL = r"runs\detect\train3\weights\best.pt"
IMAGE_SIZE = 960
CELL_DETECTOR_MODEL = r"models\cell_detector_imgs960_rectT-aug.pt"
# CELL_DETECTOR_MODEL = r"models\cell_detector_imgs960_aug.pt"  #bad
# CELL_DETECTOR_MODEL = r"runs\detect\train12\weights\best.pt"
# CELL_DETECTOR_MODEL = r"runs\detect\train54\weights\best.pt"
IMAGE_SIZE = 640
CELL_DETECTOR_MODEL = r"runs\detect\train56\weights\best.pt" # the YOLOE . bad than 960
# IMAGE_SIZE = 960
# CELL_DETECTOR_MODEL = r"runs\detect\train61\weights\best.pt" # the YOLOE

# IMAGE_SIZE = 640
# PV_DETECTOR_MODEL = r"solar_project\yolo_module_detector2\weights\best.pt"

IMAGE_SIZE = 1280
CELL_MODEL_PATH = r"models\cell_detector_best_model_8s_7_1.pt"

# cell_model = YOLO(CELL_DETECTOR_MODEL)
cell_model = YOLO(CELL_MODEL_PATH)

for img_name in image_files:
    img_path = os.path.join(train_img_dir, img_name)
    lbl_path = os.path.join(train_lbl_dir, os.path.splitext(img_name)[0] + ".txt")
    
    # img = cv2.imread(img_path)
    # poly = get_module_polygon(img)

    # if poly is not None:
    #     # Draw the polygon on the image to verify accuracy
    #     original_h , original_w = img.shape[:2]

    #     scale_factor = 640 / original_h
    #     cv2.drawContours(img, [poly], -1, (0, 255, 0), 5)
    #     cv2.imshow("Result", cv2.resize(img, (0, 0), fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_AREA))
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()
    # else:
    #     print('none', img_path)
    # continue

    pv_img = cv2.imread(img_path)
    pv_h, pv_w = pv_img.shape[:2] 
    # resized_img = cv2.resize(pv_img, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)

    # clean = preprocess_pv(resized_img, mask_threshold=10)

    # Run cell detection
    results = cell_model.predict(pv_img, conf=0.4, iou=0.6, imgsz=IMAGE_SIZE, 
                                 verbose=False
                                # save=True, project=out_img_dir, #name='visuals',
                                 )
    # continue
    boxes = results[0].boxes.xyxy.cpu().numpy()  # list of detected boxes
    boxes = [list(b) for b in boxes]  # convert to Python list if needed
    

    # print(len(boxes))
    filtered_boxes = filter_boxes_by_mode(boxes)

    pad = CELL_PADDING
    h_resized, w_resized = pv_img.shape[:2]

    for i, det in enumerate(filtered_boxes):
        x1, y1, x2, y2 = map(int, det)

        cell_crop = pv_img[y1:y2, x1:x2]
        if cell_crop.size == 0:
            continue

        cell_crop = cv2.copyMakeBorder(
            cell_crop,
            top=pad,
            bottom=pad,
            left=pad,
            right=pad,
            borderType=cv2.BORDER_REFLECT_101
        )

        cell_w_orig = x2 - x1
        cell_h_orig = y2 - y1

        crop_w = cell_w_orig + 2 * pad
        crop_h = cell_h_orig + 2 * pad

        pad_x = pad
        pad_y = pad

        cell_img_name = f"{os.path.splitext(img_name)[0]}_cell{i}.jpg"
        cell_lbl_name = f"{os.path.splitext(img_name)[0]}_cell{i}.txt"

        cv2.imwrite(os.path.join(out_img_dir, cell_img_name), cell_crop)
        # continue

        cell_labels = extract_cell_labels_from_pv(
            lbl_path,
            det,
            pv_w,
            pv_h,
            IMAGE_SIZE,  # resized width
            IMAGE_SIZE,  # resized height
            pad_x,       # x padding offset in the crop (x1 - x1_padded)
            pad_y,       # y padding offset in the crop (y1 - y1_padded)
            cell_w_orig, # original cell width (without padding, in resized image)
            cell_h_orig, # original cell height (without padding, in resized image)
            crop_w,      # padded crop width (in resized image)
            crop_h       # padded crop height (in resized image)
        )
        
        cell_lbl_file = os.path.join(out_lbl_dir, cell_lbl_name)
        if len(cell_labels) >0:
            with open(cell_lbl_file, 'w') as f:
                f.write("\n".join(cell_labels))

            cv2.imshow("cell", draw_annotation(cell_crop, i, cell_lbl_file))
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    # for i, det in enumerate(filtered_boxes):
    #     x1, y1, x2, y2 = map(int, det)
    #     # final_cell_image =  crop_and_warp_cell(pv_img, det)
    #     # if final_cell_image is not None:
    #     #     print(f"cell_{img_name}")
    #     #     # cv2.imwrite(f"cell_{img_name}", final_cell_image)
    #     #     continue
    #     # else:
    #     #     print('not none')
    #     # pv_img = draw_cell_with_pv_labels(pv_img, [x1, y1, x2, y2], i )
    #     # cv2.imshow('d', pv_img)
    #     # cv2.waitKey(0)
    #     # cv2.destroyAllWindows()

    #     # continue
    
        
    #     # Calculate padded coordinates, clamping to image boundaries
    #     x1_padded = max(0, x1 - pad)
    #     y1_padded = max(0, y1 - pad)
    #     x2_padded = min(w_resized, x2 + pad)
    #     y2_padded = min(h_resized, y2 + pad)
        
    #     # Crop with padding
    #     cell_crop = pv_img[y1_padded:y2_padded, x1_padded:x2_padded]
    #     if cell_crop.size == 0:
    #         continue
        
    #     # Original cell dimensions (without padding) in the padded crop
    #     cell_w_orig = x2 - x1
    #     cell_h_orig = y2 - y1
    #     crop_w = x2_padded - x1_padded
    #     crop_h = y2_padded - y1_padded
        
    #     # Calculate padding offset in the actual crop (may be less than pad if near image edge)
    #     pad_x = x1 - x1_padded
    #     pad_y = y1 - y1_padded
        
            # cell_labels = extract_cell_labels_from_pv(
            # lbl_path,
            # det,
            # pv_w,
            # pv_h,
            # IMAGE_SIZE,  # resized width
            # IMAGE_SIZE,  # resized height
            # pad_x,       # x padding offset in the crop (x1 - x1_padded)
            # pad_y,       # y padding offset in the crop (y1 - y1_padded)
            # cell_w_orig, # original cell width (without padding, in resized image)
            # cell_h_orig, # original cell height (without padding, in resized image)
            # crop_w,      # padded crop width (in resized image)
            # crop_h       # padded crop height (in resized image)
        # )


    #     cell_img_name = f"{os.path.splitext(img_name)[0]}_cell{i}.jpg"

    #     cv2.imwrite(os.path.join(out_img_dir, cell_img_name), cell_crop)
    
    # # cv2.imwrite(r"C:\Users\Rowan\Documents\Rowan\Yolo_test\cropp_pvs_yoloe640\\" + img_name , pv_img)
    # cv2.imshow('d', cv2.resize(pv_img, (IMAGE_SIZE, IMAGE_SIZE)))
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
