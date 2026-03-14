import cv2
import os
import shutil
import numpy as np

# ================= CONFIGURATION =================
# 1. PATHS
IMG_DIR = r"C:\Users\Rowan\Documents\Rowan\Yolo_test\pv_cropped_ready_integ\train\images"
RF_LABELS_DIR = r"C:\Users\Rowan\Documents\Rowan\Yolo_test\pv_cropped_ready_integ\train\labels"          # Ground Truth (Defects)
YOLO_LABELS_DIR = r"C:\Users\Rowan\Documents\Rowan\Yolo_test\pv_cropped_ready_integ\train\labels" # Detected Cells (candidates)
BASE_OUT = "cell_classes_yolos8_from_pv_integ/train"                       # Output folder

# 2. CLASS NAMES (Must match your data.yaml ID order)
# Important: Class 0 is usually 'examined' or 'healthy'
CLASS_NAMES = [
  "examined",
"micro_crack",
"crack",
"low_cell",
"low_string",
"isolated_area",
"contamination",
"other_error",
]

IOU_THRESHOLD = 0.3  # How much overlap is needed to call it a defect?

# ================= HELPER FUNCTIONS =================
def load_yolo(txt_path):
    """Loads YOLO labels from a txt file."""
    boxes = []
    if os.path.exists(txt_path):
        with open(txt_path, 'r') as f:
            for line in f:
                parts = list(map(float, line.strip().split()))
                boxes.append(parts)
    return boxes

def yolo_to_xyxy(box, img_w, img_h):
    """Converts YOLO (x_center, y_center, w, h) to (x1, y1, x2, y2)."""
    xc, yc, w, h = box
    x1 = int((xc - w / 2) * img_w)
    y1 = int((yc - h / 2) * img_h)
    x2 = int((xc + w / 2) * img_w)
    y2 = int((yc + h / 2) * img_h)
    # Clip to image boundaries
    return [max(0, x1), max(0, y1), min(img_w, x2), min(img_h, y2)]

def compute_iou(boxA, boxB):
    """Calculates Intersection over Union (IoU) between two boxes."""
    # Determine the coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # Compute the area of intersection rectangle
    interArea = max(0, xB - xA) * max(0, yB - yA)

    # Compute the area of both rectangles
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    # Compute IoU
    iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
    return iou

# ================= MAIN LOOP =================
# 1. Create Output Directories
# if os.path.exists(BASE_OUT):
#     shutil.rmtree(BASE_OUT)

# for name in CLASS_NAMES:
#     os.makedirs(os.path.join(BASE_OUT, name), exist_ok=True)

print(f"🚀 Starting processing...")
count = 0

# 2. Iterate through images
for img_name in os.listdir(IMG_DIR):
    if not img_name.lower().endswith(('.jpg', '.png', '.jpeg')):
        continue

    img_path = os.path.join(IMG_DIR, img_name)
    img = cv2.imread(img_path)
    if img is None: continue
    h, w, _ = img.shape

    # Define label paths
    rf_txt = os.path.join(RF_LABELS_DIR, img_name.replace(".jpg", ".txt"))
    yolo_txt = os.path.join(YOLO_LABELS_DIR, img_name.replace(".jpg", ".txt").replace(".JPG", ".txt"))

    # Skip if we don't have cell detection info (can't crop what we don't see)
    if not os.path.exists(yolo_txt):
        save_path = os.path.join(BASE_OUT, "healthy", os.path.splitext(img_name)[0]+".jpg")
        cv2.imwrite(save_path, img)
        continue

    # Load boxes
    rf_data = load_yolo(rf_txt)   # Ground Truth Defects
    cell_data = load_yolo(yolo_txt) # Detected Cells
    # Prepare Ground Truth Boxes (Defects) in pixel format
    gt_defects = []
    for d in rf_data:
        cls_id = int(d[0])
        box_coords = yolo_to_xyxy(d[1:5], w, h)
        gt_defects.append({'cls': cls_id, 'box': box_coords})
        # print(gt_defects)

    # Process each Detected Cell
    for i, cell_line in enumerate(cell_data):
        # Cell detector class is likely '0' (cell), ignore it. 
        # We only care about the box coordinates.
        cell_box = yolo_to_xyxy(cell_line[1:5], w, h)

        # DEFAULT: Assume it is HEALTHY (Class 0)
        assigned_class_id = 0

        # Check against all Ground Truth Defects
        best_iou = 0
        for defect in gt_defects:
            iou_val = compute_iou(cell_box, defect['box'])
            
            # If significant overlap, assume this cell contains that defect
            
            if iou_val > IOU_THRESHOLD:
                # If multiple defects overlap, take the one with highest IoU
                if iou_val > best_iou:
                    best_iou = iou_val
                    assigned_class_id = defect['cls']
        
        # Determine Folder Name
        if 0 <= assigned_class_id: #< len(CLASS_NAMES):
            class_folder = CLASS_NAMES[assigned_class_id]
            # print('---')
            # print(class_folder, assigned_class_id)
        else:
            class_folder = "unknown" # Safety fallback

        # CROP
        x1, y1, x2, y2 = cell_box
        crop = img[y1:y2, x1:x2]
        if crop.size == 0: continue

        # SAVE
        # Naming: OriginalImage_CellIndex.jpg
        save_filename = f"{os.path.splitext(img_name)[0]}_cell_{i}.jpg"
        save_path = os.path.join(BASE_OUT, class_folder, save_filename)
        # cv2.imwrite(save_path, crop)
        cv2.imwrite(save_path, img)
        count += 1

print(f"✅ Done! Processed {count} cell crops.")
print(f"📁 Results saved in: {BASE_OUT}")