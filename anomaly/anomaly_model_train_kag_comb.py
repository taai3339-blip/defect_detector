# ==========================================================
# 1. IMPORTS & CONFIGURATION
# ==========================================================

import os
import cv2
import random
import shutil
import glob
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix, roc_curve
from torchvision.transforms import v2
from zipfile import ZipFile
from pathlib import Path
from typing import Tuple, Optional, List
from IPython.display import FileLink

# Anomalib Imports
os.environ['TRUST_REMOTE_CODE'] = '1'
from anomalib.data import Folder
from anomalib.models import Patchcore
from anomalib.engine import Engine

# --- PATHS ---
# Adjust these paths according to your Kaggle environment
CLEAN_SOURCE = "cropped_clean_using_yo8s_1280_p10_rmvd_dirty" 
CLEAN_SOURCE2 = "c1_c2" 
DEFECT_SOURCE = r"C:\Users\Rowan\Documents\Rowan\Yolo_test\cells_by_class\crack"

OUTPUT_ROOT = "pv_organized_dataset_crack"
ANOMALIB_ROOT = "anomalib_dataset"
EXPORT_DIR = "patchcore_model_robust_export"

# --- SETTINGS ---
TRAIN_RATIO = 0.7
TARGET_IMG_SIZE = 320
USE_CLAHE = False  # Based on notebook usage
TRAIN_BATCH_SIZE = 16
EVAL_BATCH_SIZE = 16
NUM_WORKERS = 4
EPOCHS = 1

# --- BAD CONTRAST LIST ---
bad_contrast = '''DSC_1493_cell25.jpg
DSC_0791_cell39.jpg
DSC_1424_cell68.jpg
DSC_3043_cell45.jpg
DSC_0948_cell42.jpg
DSC_0833_cell28.jpg
DSC_2000_cell43.jpg
DSC_0833_cell114.jpg
DSC_2251_cell89.jpg
DSC_0848_cell31.jpg
DSC_1502_cell52.jpg
DSC_1614_cell134.jpg
DSC_0848_cell271.jpg
DSC_0833_cell15.jpg
DSC_1823_cell11.jpg
DSC_0833_cell47.jpg
DSC_2066_cell56.jpg
DSC_0496_cell40.jpg
DSC_1315_cell100.jpg
DSC_2251_cell118.jpg
DSC_1424_cell19.jpg
DSC_1614_cell26.jpg
DSC_1315_cell55.jpg
DSC_2251_cell95.jpg
DSC_1823_cell112.jpg
DSC_1335_cell116.jpg
DSC_1823_cell102.jpg
DSC_1424_cell93.jpg
DSC_0496_cell46.jpg
DSC_1335_cell43.jpg
DSC_1335_cell121.jpg
DSC_0496_cell35.jpg
DSC_1614_cell63.jpg
DSC_0833_cell108.jpg
DSC_1424_cell56.jpg
DSC_1335_cell38.jpg
DSC_1424_cell99.jpg
DSC_0833_cell93.jpg
DSC_1994_cell44.jpg
DSC_0791_cell268.jpg
DSC_1994_cell25.jpg
DSC_1614_cell101.jpg
DSC_2251_cell131.jpg
DSC_0848_cell182.jpg
DSC_0848_cell186.jpg
DSC_0848_cell55.jpg
DSC_0833_cell63.jpg
DSC_1456_cell29.jpg
DSC_2251_cell114.jpg
DSC_2000_cell137.jpg
DSC_2066_cell127.jpg
DSC_2066_cell50.jpg
DSC_1456_cell12.jpg
DSC_1424_cell42.jpg
DSC_2066_cell96.jpg
DSC_2066_cell123.jpg
DSC_0678_cell51.jpg
DSC_1614_cell95.jpg
DSC_0848_cell13.jpg
DSC_2000_cell127.jpg
452_jpg.rf.9bb991efb724acb58e71f32b67898294_cell85.jpg
DSC_1424_cell86.jpgDSC_1335_cell79.jpg
DSC_1424_cell51.jpg
DSC_0848_cell81.jpg
DSC_0496_cell30.jpg
DSC_1335_cell15.jpg
DSC_2066_cell93.jpg
DSC_1424_cell59.jpg
DSC_0531_cell16.jpg
DSC_0678_cell21.jpg
DSC_1614_cell98.jpg
DSC_2066_cell130.jpg
DSC_0496_cell38.jpg
DSC_0496_cell59.jpg
DSC_0678_cell45.jpg
DSC_1823_cell37.jpg
DSC_3103_cell91.jpg
DSC_2066_cell113.jpg
DSC_2000_cell123.jpg
DSC_0791_cell85.jpg
DSC_1823_cell48.jpg
DSC_0791_cell232.jpg
DSC_1335_cell52.jpg
DSC_2251_cell78.jpg
DSC_0948_cell16.jpg
DSC_1493_cell82.jpg
DSC_1424_cell12.jpg
DSC_2242_cell124.jpg
DSC_0833_cell101.jpg
DSC_1614_cell37.jpg
DSC_1335_cell46.jpg
DSC_0833_cell53.jpg
DSC_3043_cell133.jpg
DSC_1823_cell99.jpg
DSC_0791_cell199.jpg
DSC_2066_cell53.jpg
DSC_0678_cell17.jpg
DSC_0791_cell194.jpg
DSC_1315_cell15.jpg
DSC_1424_cell29.jpg
DSC_1823_cell118.jpg
DSC_1493_cell95.jpg
DSC_1335_cell55.jpg
DSC_2251_cell84.jpg
DSC_0948_cell37.jpg
DSC_0678_cell27.jpg
DSC_2000_cell46.jpg
DSC_2251_cell102.jpg
DSC_0848_cell156.jpg
DSC_1493_cell59.jpg
DSC_1823_cell120.jpg
DSC_0848_cell19.jpg
DSC_1315_cell135.jpg
DSC_0848_cell40.jpg
DSC_0496_cell63.jpg
DSC_1579_cell104.jpg
DSC_1424_cell96.jpg
DSC_0833_cell146.jpg
DSC_0848_cell105.jpg
DSC_0496_cell52.jpg
DSC_1424_cell82.jpg
DSC_1456_cell8.jpg
DSC_0848_cell91.jpg
DSC_2000_cell105.jpg
DSC_0791_cell89.jpg
DSC_0791_cell225.jpg
DSC_1424_cell47.jpg
DSC_1493_cell41.jpg
DSC_0848_cell198.jpg
DSC_3043_cell86.jpg
DSC_3043_cell12.jpg
DSC_1994_cell30.jpg
DSC_2251_cell99.jpg
DSC_1493_cell47.jpg
DSC_2000_cell40.jpg
DSC_0833_cell67.jpg
DSC_1493_cell77.jpg
DSC_0833_cell77.jpg
DSC_2000_cell130.jpg
DSC_0791_cell170.jpg
DSC_1493_cell33.jpg
DSC_0833_cell121.jpg
DSC_0678_cell33.jpg
DSC_1994_cell36.jpg
DSC_0678_cell38.jpg
DSC_1614_cell7.jpg
DSC_3043_cell39.jpg
DSC_0791_cell68.jpg
DSC_1493_cell98.jpg
DSC_0848_cell230.jpg
DSC_1424_cell35.jpg
DSC_1994_cell8.jpg
DSC_1502_cell100.jpg
DSC_0848_cell67.jpg
DSC_1424_cell89.jpg
DSC_1994_cell22.jpg
DSC_0848_cell127.jpg
DSC_0791_cell187.jpg
DSC_3174_cell25.jpg
DSC_3227_cell104.jpg
DSC_3227_cell58.jpg
DSC_3227_cell10.jpg
DSC_3185_cell88.jpg
DSC_3248_cell80.jpg
DSC_3227_cell45.jpg
DSC_3174_cell102.jpg
DSC_3329_cell117.jpg
DSC_3237_cell81.jpg
DSC_3227_cell85.jpg
DSC_3339_cell85.jpg
DSC_3174_cell120.jpg
DSC_3195_cell74.jpg
DSC_3329_cell28.jpg
DSC_3349_cell132.jpg
DSC_3329_cell56.jpg
DSC_3339_cell15.jpg
DSC_3174_cell45.jpg
DSC_3329_cell87.jpg
DSC_3339_cell60.jpg
DSC_3227_cell128.jpg
DSC_3329_cell135.jpg
DSC_3195_cell77.jpg
DSC_3349_cell108.jpg
DSC_3329_cell32.jpg
DSC_3237_cell49.jpg
DSC_2242_cell52.jpg
DSC_3349_cell25.jpg
DSC_1424_cell27.jpg
DSC_3227_cell91.jpg
DSC_1424_cell78.jpg
DSC_3216_cell4.jpg
DSC_3258_cell63.jpg
DSC_1823_cell101.jpg
DSC_0833_cell12.jpg
DSC_0848_cell52.jpg
DSC_0791_cell102.jpg
DSC_2000_cell45.jpg
DSC_2066_cell51.jpg
DSC_1493_cell34.jpg
DSC_3359_cell81.jpg
DSC_2242_cell135.jpg
DSC_1994_cell33.jpg
DSC_1493_cell91.jpg
45-91-_jpg.rf.de9fe61de99733287d409acc88af4e55_cell_70.jpg
DSC_0678_cell34.jpg
DSC_0496_cell36.jpg
DSC_0848_cell259.jpg
DSC_2000_cell80.jpg
DSC_0948_cell4.jpg
DSC_0848_cell211.jpg
DSC_2066_cell18.jpg
DSC_1493_cell90.jpg
DSC_1424_cell39.jpg
DSC_1335_cell61.jpg
DSC_0791_cell118.jpg
DSC_2251_cell93.jpg
DSC_2000_cell106.jpg
DSC_1456_cell3.jpg
DSC_2251_cell101.jpg
DSC_0791_cell51.jpg
DSC_0848_cell129.jpg
DSC_1424_cell14.jpg
DSC_1424_cell21.jpg
DSC_1579_cell118.jpg
DSC_1493_cell18.jpg
DSC_1424_cell95.jpg
DSC_0678_cell8.jpg
DSC_0848_cell79.jpg
DSC_1579_cell99.jpg
DSC_1493_cell11.jpg
DSC_3174_cell50.jpg
572_jpg.rf.da4d6c1fb8d7c3339a7175a2e59e53b6_cell82.jpg
DSC_1994_cell15.jpg
DSC_3103_cell88.jpg
DSC_0948_cell36.jpg
DSC_1424_cell90.jpg
DSC_0791_cell162.jpg
DSC_2066_cell103.jpg
DSC_0791_cell107.jpg
DSC_2000_cell19.jpg
DSC_2066_cell31.jpg
DSC_1493_cell76.jpg
DSC_3174_cell43.jpg
DSC_0833_cell117.jpg
DSC_1335_cell82.jpg
DSC_1994_cell12.jpg
DSC_0678_cell50.jpg
DSC_0833_cell102.jpg
DSC_0833_cell139.jpg
DSC_0791_cell266.jpg
DSC_2000_cell41.jpg
DSC_0791_cell67.jpg
DSC_0948_cell47.jpg
DSC_2066_cell99.jpg
DSC_0791_cell242.jpg
DSC_1424_cell86.jpg
DSC_2242_cell93.jpg
DSC_2066_cell124.jpg
DSC_0791_cell218.jpg
DSC_0496_cell51.jpg
DSC_0496_cell21.jpg
DSC_0791_cell113.jpg
DSC_1994_cell39.jpg
DSC_0791_cell40.jpg
DSC_1335_cell112.jpg
DSC_0848_cell223.jpg
DSC_1424_cell43.jpg
DSC_0791_cell182.jpg
DSC_0791_cell74.jpg
DSC_2066_cell110.jpg
DSC_2066_cell135.jpg
DSC_1424_cell31.jpg
DSC_2066_cell57.jpg
DSC_0848_cell158.jpg
DSC_3103_cell100.jpg
DSC_1335_cell122.jpg
DSC_3174_cell114.jpg
DSC_0791_cell71.jpg
DSC_1456_cell7.jpg
DSC_2000_cell57.jpg
DSC_0833_cell128.jpg
DSC_1493_cell81.jpg
DSC_2251_cell90.jpg
DSC_2066_cell80.jpg
DSC_0833_cell60.jpg
DSC_2242_cell105.jpg
DSC_0848_cell33.jpg
DSC_0496_cell58.jpg
DSC_0833_cell39.jpg
DSC_3174_cell60.jpg
DSC_0948_cell50.jpg
DSC_1335_cell104.jpg
DSC_1335_cell100.jpg
DSC_0833_cell25.jpg
DSC_1335_cell97.jpg
DSC_3174_cell63.jpg
DSC_0791_cell32.jpg
DSC_2000_cell93.jpg
DSC_2000_cell87.jpg
DSC_3174_cell21.jpg
DSC_3043_cell97.jpg
DSC_1424_cell63.jpg
DSC_1579_cell45.jpg
DSC_1614_cell32.jpg
DSC_1424_cell49.jpg
DSC_3103_cell98.jpg
DSC_0678_cell20.jpg
331_jpg.rf.6eaee0d2314aca27614d0a4141de30d8_cell37.jpg
DSC_1273_cell20.jpg
DSC_1994_cell7.jpg
DSC_0848_cell265.jpg
DSC_1493_cell53.jpg
DSC_0948_cell43.jpg
DSC_2066_cell119.jpg
DSC_1424_cell36.jpg
DSC_1424_cell6.jpg
DSC_0848_cell43.jpg
DSC_0848_cell63.jpg
DSC_1424_cell53.jpg
DSC_1994_cell24.jpg
DSC_2066_cell63.jpg
DSC_0678_cell43.jpg
DSC_0791_cell101.jpg
DSC_0678_cell37.jpg
DSC_0833_cell75.jpg
DSC_0496_cell62.jpg
DSC_1493_cell39.jpg
DSC_1335_cell57.jpg
538_jpg.rf.d6e74749970de5ba472222d8a2efa479_cell54.jpg
DSC_3227_cell94.jpg
DSC_3339_cell79.jpg
DSC_3227_cell79.jpg
DSC_3268_cell32.jpg
DSC_3237_cell96.jpg
DSC_3227_cell74.jpg
DSC_3227_cell116.jpg
DSC_3185_cell116.jpg
DSC_3349_cell9.jpg
DSC_3216_cell60.jpg
DSC_3227_cell56.jpg
DSC_3349_cell113.jpg
DSC_3185_cell26.jpg
DSC_3339_cell27.jpg
DSC_3329_cell86.jpg
DSC_3258_cell128.jpg
DSC_3359_cell115.jpg
DSC_3185_cell104.jpg
DSC_3268_cell43.jpg
DSC_3216_cell89.jpg
DSC_3227_cell106.jpg
DSC_3227_cell97.jpg
DSC_0791_cell97.jpg
DSC_0848_cell93.jpg
DSC_2251_cell38.jpg
DSC_2000_cell15.jpg
DSC_3359_cell91.jpg
DSC_0791_cell219.jpg
DSC_0791_cell44.jpg
DSC_0948_cell38.jpg
DSC_0496_cell53.jpg
DSC_0678_cell22.jpg
DSC_1493_cell12.jpg
DSC_0791_cell259.jpg
DSC_3227_cell88.jpg
DSC_3174_cell36.jpg
DSC_3185_cell56.jpg
DSC_2000_cell97.jpg
DSC_0848_cell272.jpg
45-91-_jpg.rf.de9fe61de99733287d409acc88af4e55_cell_70.jpg'''.split('\n')


# ==========================================================
# 2. PREPROCESSING CLASS
# ==========================================================

class RobustPreprocessor:
    """
    Centralized preprocessing pipeline: Validation, CLAHE, Resize with Padding.
    """
    def __init__(self, target_size: int = 320, use_clahe: bool = True):
        self.target_size = target_size
        self.use_clahe = use_clahe
        if use_clahe:
            self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    def validate_image(self, img: np.ndarray) -> Tuple[bool, str]:
        if img is None: return False, "Failed to load image"
        if len(img.shape) < 2: return False, "Invalid dimensions"
        
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
        mean_brightness = np.mean(img)
        std_brightness = np.std(img)
        
        if mean_brightness < 10: return False, "Too dark"
        if mean_brightness > 245: return False, "Overexposed"
        if std_brightness < 5: return False, "Low contrast"
        
        return True, "OK"

    def resize_with_padding(self, img: np.ndarray) -> np.ndarray:
        """Resize maintaining aspect ratio using BORDER_REPLICATE padding."""
        h, w = img.shape[:2]
        
        # Calculate padding to make square
        if h > w:
            diff = h - w
            pad_top, pad_bottom = 0, 0
            pad_left = diff // 2
            pad_right = diff - pad_left
        else:
            diff = w - h
            pad_top = diff // 2
            pad_bottom = diff - pad_top
            pad_left, pad_right = 0, 0
            
        # Pad
        padded = cv2.copyMakeBorder(img, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_REPLICATE)
        
        # Resize to target
        target = (self.target_size, self.target_size) if isinstance(self.target_size, int) else self.target_size
        return cv2.resize(padded, target, interpolation=cv2.INTER_CUBIC)

    def process(self, image_path: str) -> Tuple[Optional[np.ndarray], dict]:
        """
        Main preprocessing pipeline entry point.
        Returns: (processed_image_rgb, metadata_dict)
        """
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return None, {'is_valid': False, 'reason': 'load_failed'}
            
        valid, msg = self.validate_image(img)
        if not valid:
            return None, {'is_valid': False, 'reason': msg}

        if self.use_clahe:
            img = self.clahe.apply(img)
            
        img = self.resize_with_padding(img)
        
        # Convert to RGB for model compatibility
        img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        return img_rgb, {'is_valid': True}


# ==========================================================
# 3. DATA ORGANIZATION FUNCTIONS
# ==========================================================

def organize_raw_dataset():
    """Step 1: Copy raw files to OUTPUT_ROOT, excluding bad contrast."""
    print(f"--- Step 1: Organizing Raw Dataset to {OUTPUT_ROOT} ---")
    
    # Define folders
    train_good_dir = os.path.join(OUTPUT_ROOT, "train", "good")
    test_good_dir = os.path.join(OUTPUT_ROOT, "test", "good")
    test_defect_dir = os.path.join(OUTPUT_ROOT, "test", "defect")
    
    for folder in [train_good_dir, test_good_dir, test_defect_dir]:
        if os.path.exists(folder): shutil.rmtree(folder)
        os.makedirs(folder, exist_ok=True)
        
    # Gather Clean Files
    clean_files = []
    valid_exts = ('.jpg', '.jpeg', '.png', '.bmp')
    
    for source in [CLEAN_SOURCE, CLEAN_SOURCE2]:
        if os.path.exists(source):
            for f in os.listdir(source):
                if f.lower().endswith(valid_exts):
                    if f in bad_contrast: continue
                    clean_files.append(os.path.join(source, f))
    
    random.shuffle(clean_files)
    split_idx = int(len(clean_files) * TRAIN_RATIO)
    
    train_clean = clean_files[:split_idx]
    test_clean = clean_files[split_idx:]
    
    # Copy Clean
    for f in train_clean: shutil.copy(f, os.path.join(train_good_dir, os.path.basename(f)))
    for f in test_clean: shutil.copy(f, os.path.join(test_good_dir, os.path.basename(f)))
    
    # Gather Defect Files
    defect_files = []
    if os.path.exists(DEFECT_SOURCE):
        for f in os.listdir(DEFECT_SOURCE):
            if f.lower().endswith(valid_exts):
                defect_files.append(os.path.join(DEFECT_SOURCE, f))
    
    random.shuffle(defect_files)
    for f in defect_files: shutil.copy(f, os.path.join(test_defect_dir, os.path.basename(f)))
    
    print(f"  Train Clean: {len(train_clean)}")
    print(f"  Test Clean:   {len(test_clean)}")
    print(f"  Test Defect:  {len(defect_files)}")


def preprocess_and_save_for_anomalib():
    """Step 2: Apply centralized preprocessing and save to ANOMALIB_ROOT."""
    print(f"\n--- Step 2: Preprocessing & Saving to {ANOMALIB_ROOT} ---")
    
    if os.path.exists(ANOMALIB_ROOT): shutil.rmtree(ANOMALIB_ROOT)
    
    # Create Anomalib structure
    # /anomalib_dataset/train/good
    # /anomalib_dataset/test/good
    # /anomalib_dataset/test/defect
    os.makedirs(f"{ANOMALIB_ROOT}/train/good", exist_ok=True)
    os.makedirs(f"{ANOMALIB_ROOT}/test/good", exist_ok=True)
    os.makedirs(f"{ANOMALIB_ROOT}/test/defect", exist_ok=True)
    
    preprocessor = RobustPreprocessor(target_size=TARGET_IMG_SIZE, use_clahe=USE_CLAHE)
    
    # Helper to process a list of files and save to destination
    def process_and_save(file_list, save_dir):
        count = 0
        for i, img_path in enumerate(file_list):
            img, meta = preprocessor.process(img_path)
            if meta['is_valid']:
                cv2.imwrite(os.path.join(save_dir, f"img_{i:04d}.jpg"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                count += 1
        return count

    # 1. Process Training Data (Combine Train Good + Validation Good from raw split)
    # Note: The notebook combined train/val from raw source into the 'train' set for Anomalib.
    raw_train_good = glob.glob(f"{OUTPUT_ROOT}/train/good/*")
    # We assume the 'test/good' from raw split contains the hold-out set, 
    # but the notebook logic created a 'val' folder in SUBSET_DATA_ROOT then combined it.
    # To replicate exactly: The notebook used a SUBSET_DATA_ROOT with train/val/test splits.
    # Here we will use OUTPUT_ROOT which has train/test.
    # Let's split OUTPUT_ROOT/test/good into Val/Test for the final Anomalib structure if we want exact replication,
    # OR just treat OUTPUT_ROOT/train/good as train and part of OUTPUT_ROOT/test/good as test.
    
    # Simplified flow matching the notebook's intent:
    # Use all of OUTPUT_ROOT/train/good for training.
    # Use OUTPUT_ROOT/test/good for testing (normal).
    
    train_files = glob.glob(f"{OUTPUT_ROOT}/train/good/*")
    test_good_files = glob.glob(f"{OUTPUT_ROOT}/test/good/*")
    test_defect_files = glob.glob(f"{OUTPUT_ROOT}/test/defect/*")
    
    n_train = process_and_save(train_files, f"{ANOMALIB_ROOT}/train/good")
    n_test_good = process_and_save(test_good_files, f"{ANOMALIB_ROOT}/test/good")
    n_test_defect = process_and_save(test_defect_files, f"{ANOMALIB_ROOT}/test/defect")
    
    print(f"  Preprocessed Train: {n_train}")
    print(f"  Preprocessed Test Good: {n_test_good}")
    print(f"  Preprocessed Test Defect: {n_test_defect}")


def compute_normalization_stats():
    """Step 3: Compute Mean/Std from preprocessed training images."""
    print("\n--- Step 3: Computing Normalization Stats ---")
    
    image_paths = glob.glob(f"{ANOMALIB_ROOT}/train/good/*.jpg")
    print(f"  Analyzing {len(image_paths)} images...")
    
    mean = torch.zeros(3)
    std = torch.zeros(3)
    transform = v2.ToImage()
    
    for i, path in enumerate(image_paths):
        try:
            img = Image.open(path).convert('RGB')
            img_tensor = transform(img) # uint8
            img_tensor = img_tensor.float() / 255.0
            
            mean += img_tensor.mean(dim=(1, 2))
            std += img_tensor.std(dim=(1, 2))
        except Exception as e:
            print(f"Error reading {path}: {e}")
            
    mean /= len(image_paths)
    std /= len(image_paths)
    
    print(f"  Mean:  {mean.tolist()}")
    print(f"  Std:   {std.tolist()}")
    return mean.tolist(), std.tolist()

from PIL import Image

# ==========================================================
# 4. MAIN PIPELINE
# ==========================================================

def main():
    # 1. Organize Raw Files
    organize_raw_dataset()
    
    # 2. Centralized Preprocessing (Applied to ALL images once)
    preprocess_and_save_for_anomalib()
    
    # 3. Compute Stats
    try:
        mean, std = compute_normalization_stats()
    except Exception as e:
        print(f"Error computing stats, using defaults: {e}")
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

    # 4. Define Transforms
    train_transform = v2.Compose([
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandomVerticalFlip(p=0.5),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=mean, std=std)
    ])

    eval_transform = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=mean, std=std)
    ])

    # 5. Setup Datamodule
    # Note: normal_dir points to train/good. 
    # val_split_mode="from_test" will take a portion of normal_dir for validation.
    datamodule = Folder(
        name="solar_panel_robust",
        root=ANOMALIB_ROOT,
        normal_dir="train/good",
        normal_test_dir="test/good",
        abnormal_dir="test/defect",
        train_batch_size=TRAIN_BATCH_SIZE,
        eval_batch_size=EVAL_BATCH_SIZE,
        num_workers=NUM_WORKERS,
        val_split_mode="from_test",
        val_split_ratio=0.2, # Use 20% of train for validation
        seed=42,
    )
    datamodule.train_transform = train_transform
    datamodule.eval_transform = eval_transform

    # 6. Setup Model
    model = Patchcore(
        backbone="resnet18",
        layers=["layer2", "layer3"],
        pre_trained=True,
        coreset_sampling_ratio=0.01,
        num_neighbors=9,
    )

    # 7. Setup Engine
    engine = Engine(
        max_epochs=EPOCHS,
        accelerator="auto",
        devices=1,
        enable_progress_bar=False,
        default_root_dir=EXPORT_DIR,
    )

    # 8. Train
    print("\n--- Step 4: Training ---")
    engine.fit(model=model, datamodule=datamodule)

    # 9. Test
    print("\n--- Step 5: Testing ---")
    test_results = engine.test(model=model, datamodule=datamodule)

    # 10. Export
    print("\n--- Step 6: Exporting Model ---")
    engine.export(model=model, export_type="torch", export_root=EXPORT_DIR)

    # 11. Evaluation Metrics (Manual Calculation)
    print("\n--- Step 7: Detailed Evaluation ---")
    predictions = engine.predict(model, datamodule=datamodule)
    
    y_true, y_scores = [], []
    for batch in predictions:
        y_true.extend(batch["gt_label"].cpu().numpy())
        # Average anomaly map score per image
        batch_scores = batch["anomaly_map"].mean(dim=[1, 2]).cpu().numpy()
        y_scores.extend(batch_scores)

    # ROC & AUC
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    print(f"AUROC: {roc_auc:.4f}")

    # Best Threshold (Youden's J statistic)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    print(f"Optimal Threshold: {optimal_threshold:.4f}")

    # Confusion Matrix
    y_pred = (np.array(y_scores) > optimal_threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:")
    print(cm)

    # Plotting
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(fpr, tpr, label=f"ROC (AUC = {roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title("ROC Curve"); plt.legend()

    plt.subplot(1, 2, 2)
    normal_scores = np.array(y_scores)[np.array(y_true)==0]
    defect_scores = np.array(y_scores)[np.array(y_true)==1]
    plt.hist(normal_scores, bins=30, alpha=0.5, label='Normal')
    plt.hist(defect_scores, bins=30, alpha=0.5, label='Defect')
    plt.axvline(optimal_threshold, color='red', linestyle='--')
    plt.legend(); plt.title("Score Distribution")
    plt.show()

    # 12. Zip Weights
    print("\n--- Step 8: Zipping Model ---")
    weights_dir = os.path.join(EXPORT_DIR, "weights", "torch")
    zip_path = "patchcore_model_export.zip"
    if os.path.exists(weights_dir):
        with ZipFile(zip_path, 'w') as zipf:
            for root, dirs, files in os.walk(weights_dir):
                for file in files:
                    if file.endswith('.pt') or file.endswith('.ckpt'):
                        zipf.write(os.path.join(root, file), os.path.relpath(os.path.join(root, file), weights_dir))
        print(f"Model zipped to: {zip_path}")
        display(FileLink(zip_path))
    else:
        print("Weights directory not found.")

from sklearn.metrics import auc

if __name__ == "__main__":
    main()