

import os
import shutil
import cv2
import glob
import random
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision.transforms import v2
from anomalib.data import Folder
from anomalib.models import Patchcore
from anomalib.engine import Engine
from anomalib import TaskType

# --- CONFIGURATION ---
# Where your 40 clean images are currently stored
RAW_IMAGES_DIR = r"C:\Users\Rowan\Documents\Rowan\all clean" 

# Where we will build the training dataset
DATASET_DIR = "./dataset/solar_panel_clean_anomaly"
RESULTS_DIR = "./results"

# Fix for common Windows error with Intel libraries
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def setup_directories_and_data():
    """
    Takes your raw images, splits them into Train/Test, 
    and generates fake 'defects' so we can test the model.
    """
    if os.path.exists(DATASET_DIR):
        shutil.rmtree(DATASET_DIR)
    
    # Create required subfolders
    os.makedirs(f"{DATASET_DIR}/train/good", exist_ok=True)
    os.makedirs(f"{DATASET_DIR}/test/good", exist_ok=True)
    os.makedirs(f"{DATASET_DIR}/test/defect", exist_ok=True)

    # Get list of images
    image_paths = glob.glob(f"{RAW_IMAGES_DIR}/*")
    valid_exts = ['.jpg', '.jpeg', '.png', '.bmp']
    image_paths = [p for p in image_paths if os.path.splitext(p)[1].lower() in valid_exts]

    if not image_paths:
        print(f"ERROR: No images found in {RAW_IMAGES_DIR}. Please put your images there!")
        return False

    print(f"Found {len(image_paths)} images. Preparing dataset...")

    # Split: Keep 5 for testing, use rest for training
    random.shuffle(image_paths)
    test_count = 5
    train_imgs = image_paths[test_count:]
    test_imgs = image_paths[:test_count]

    # 1. Move Train Images
    for p in train_imgs:
        shutil.copy(p, f"{DATASET_DIR}/train/good/")

    # 2. Move Test Good Images (Clean ones for validation)
    for p in test_imgs[:2]: # Take 2 clean ones for test
        shutil.copy(p, f"{DATASET_DIR}/test/good/")

    # 3. Create Synthetic Defects (Fake Cracks) on the other test images
    # We do this because you don't have real broken panel images yet.
    print("Generating synthetic defects for testing...")
    for i, p in enumerate(test_imgs[2:]):
        img = cv2.imread(p)
        if img is None: continue
        
        # Draw random white lines (simulating cracks)
        for _ in range(random.randint(1, 3)):
            pt1 = (random.randint(0, img.shape[1]), random.randint(0, img.shape[0]))
            pt2 = (random.randint(0, img.shape[1]), random.randint(0, img.shape[0]))
            cv2.line(img, pt1, pt2, (200, 200, 200), thickness=2)
            
        cv2.imwrite(f"{DATASET_DIR}/test/defect/synthetic_{i}.jpg", img)

    return True

def run_training():
    # 1. Prepare Data
    if not setup_directories_and_data():
        return

    # 2. Configure Transforms (Resize to 256x256)
    transform = v2.Compose([
        v2.Resize((256, 256), antialias=True),
        v2.ToImage(), 
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 3. Setup Dataset
    print("Loading Data...")
    datamodule = Folder(
        name="solar_panel",
        root=DATASET_DIR,
        normal_dir="good",
        abnormal_dir="defect",
        # task="classification",
        train_batch_size=4,
        eval_batch_size=4,
        num_workers=0, # Must be 0 on Windows usually
        # train_transform=transform,
        # eval_transform=transform
    )

    # 4. Setup Model (PatchCore)
    print("Initializing Model...")
    model = Patchcore(
        backbone="resnet18",
        pre_trained=True,
    )

    # 5. Setup Engine
    engine = Engine(
        task=TaskType.CLASSIFICATION,
        max_epochs=1,
        accelerator="auto", # Will pick GPU if available, else CPU
        devices=1,
        default_root_dir=RESULTS_DIR
    )

    # 6. Train & Test
    print("Starting Training...")
    engine.fit(datamodule=datamodule, model=model)
    
    print("Running Validation...")
    engine.test(datamodule=datamodule,model=model)
    
    print(f"\nSUCCESS! Results saved in {RESULTS_DIR}")
    print("Look inside the 'results' folder for the images with heatmaps.")

if __name__ == "__main__":
    run_training()