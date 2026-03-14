import os
from pathlib import Path
import shutil

DATASET_DIR = r"C:\Users\Rowan\Documents\Rowan\Yolo_test\pv_cropped_ready_integ"
SPLITS = ["train", "valid", "test"]
IMAGE_EXTS = [".jpg", ".jpeg", ".png"]

out_dir = r"C:\Users\Rowan\Documents\Rowan\Yolo_test\anomaly\some_healthy_from_mixed_after_yolo8s\\"

os.makedirs(out_dir, exist_ok=True)


unlabeled_images = []

for split in SPLITS:
    img_dir = Path(DATASET_DIR) / split / "images"
    lbl_dir = Path(DATASET_DIR) / split / "labels"

    for img_path in img_dir.iterdir():
        if img_path.suffix.lower() not in IMAGE_EXTS:
            continue

        label_path = lbl_dir / (img_path.stem + ".txt")

        # Case 1: label file missing
        if not label_path.exists():
            unlabeled_images.append(str(img_path))
            continue

        # Case 2: label file exists but empty
        if label_path.stat().st_size == 0:
            unlabeled_images.append(str(img_path))

print(f"Found {len(unlabeled_images)} unlabeled images:\n")

for img in unlabeled_images:
    # print(img)
    img_n = os.path.basename(img)
    # print(img_n)
    # break
    shutil.copy(img_dir / img, out_dir+img_n)