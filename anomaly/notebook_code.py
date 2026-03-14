# # TOP OF YOUR SCRIPT - First lines
# import subprocess
# import sys
# import os

# # Clean install numpy with specific version
# subprocess.run([sys.executable, "-m", "pip", "install", "numpy==1.24.4", "--upgrade", "--force-reinstall", "--no-deps"], 
#                capture_output=True, check=False)

# # Clear any cached imports
# sys.path.insert(0, '/usr/local/lib/python3.12/dist-packages')

# # Now import your libraries
# import numpy as np
# print(f"NumPy version after reinstall: {np.__version__}")

# --- CELL 1: THE CLEAN SLATE INSTALL ---

# 1. UNINSTALL everything that creates the conflict.
# We remove Kaggle's pre-installed Torch so we can install a compatible one.
# !pip uninstall -y -q torch torchvision torchaudio numpy anomalib lightning
# !pip uninstall  -y -q numpy
# 2. INSTALL a clean, compatible stack from PyPI.
# We install standard PyTorch (which works with Numpy 1.26) and Anomalib together.
# !pip install -q "numpy==1.24.4" #"torch>=2" "torchvision>=0.17" "anomalib[full]==1.1.1" "lightning>=2.2.0"

# !pip install -q "numpy==1.26.4" 

# !pip install --only-binary :all: --no-cache-dir numpy==1.26.4

# pip install opencv-python==4.9.0.80 opencv-python-headless==4.9.0.80

!pip install "anomalib"



# !pip install "lightning>=2.2.0"


# # 3. VERIFY imports immediately (to catch issues early)
# # import torch
# import numpy as np
# import anomalib
# import lightning

# print(f"Successfully installed:")
# print(f" - Numpy: {np.__version__}")
# print(f" - Torch: {torch.__version__}")
# print(f" - Anomalib: {anomalib.__version__}")

bad_contrast='''DSC_1493_cell25.jpg
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

import os
import shutil
import random
from pathlib import Path
import cv2

# ==========================================================
# USER CONFIGURATION
# ==========================================================
# CHANGE THESE PATHS to point to your existing folders in Kaggle
# Example: "/kaggle/input/my-dataset/clean"
CLEAN_SOURCE = "/kaggle/input/mixed-and-clean-anomaly/cropped_clean_using_yo8s_1280_p10_rmvd_dirty/cropped_clean_using_yo8s_1280_p10_rmvd_dircty" 
CLEAN_SOURCE2 = "/kaggle/input/mixed-and-clean-anomaly/mixed_crops/c1_c2" 

DEFECT_SOURCE = "/kaggle/input/test-anomaly-cracks/crack"

# Where we will create the new organized dataset
OUTPUT_ROOT = "/kaggle/working/pv_organized_dataset_crack"

# Split ratio: 70% Clean for Training, 20% Clean for Validation
TRAIN_RATIO = 0.7
# IMG_SIZE = 320 # 256

def organize_dataset():
    print(f"--- Organizing Dataset ---")
    
    # 1. Create Directory Structure
    # We need:
    # /pv_organized_dataset/train/good/
    # /pv_organized_dataset/test/good/
    # /pv_organized_dataset/test/defect/
    
    train_good_dir = os.path.join(OUTPUT_ROOT, "train", "good")
    test_good_dir = os.path.join(OUTPUT_ROOT, "test", "good")
    test_defect_dir = os.path.join(OUTPUT_ROOT, "test", "defect")
    
    for folder in [train_good_dir, test_good_dir, test_defect_dir]:
        if os.path.exists(folder):
            shutil.rmtree(folder)
        os.makedirs(folder, exist_ok=True)
        
    # 2. Process Clean Images (Split into Train/Test)
    clean_files = []
    valid_exts = ('.jpg', '.jpeg', '.png', '.bmp')
    
    if os.path.exists(CLEAN_SOURCE):
        for f in os.listdir(CLEAN_SOURCE):
            if f.lower().endswith(valid_exts):
                if f in bad_contrast:
                    continue
                clean_files.append(os.path.join(CLEAN_SOURCE, f))
    else:
        print(f"Warning: Clean source path not found: {CLEAN_SOURCE}")

    if os.path.exists(CLEAN_SOURCE2):
        for f in os.listdir(CLEAN_SOURCE2):
            if f.lower().endswith(valid_exts):
                if f in bad_contrast:
                    continue
                clean_files.append(os.path.join(CLEAN_SOURCE2, f))
    else:
        print(f"Warning: Clean source path not found: {CLEAN_SOURCE2}")

    print(f"Found {len(clean_files)} clean images.")
    
    random.shuffle(clean_files)
    
    split_idx = int(len(clean_files) * TRAIN_RATIO)
    
    train_clean = clean_files[:split_idx]
    test_clean = clean_files[split_idx:]
    
    # Copy Train Clean
    for f in train_clean:
        # img = cv2.imread(f)
        # img = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation= cv2.INTER_CUBIC)
        # cv2.imwrite(os.path.join(train_good_dir, os.path.basename(f)), img)
        
        shutil.copy(f, os.path.join(train_good_dir, os.path.basename(f)))
        
    # Copy Test Clean (for Thresholding/FPR)
    for f in test_clean:
        # img = cv2.imread(f)
        # img = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation= cv2.INTER_CUBIC)
        # cv2.imwrite(os.path.join(test_good_dir, os.path.basename(f)), img)
        
        shutil.copy(f, os.path.join(test_good_dir, os.path.basename(f)))
        
    # 3. Process Defect Images (All go to Test/Defect)
    defect_files = []
    if os.path.exists(DEFECT_SOURCE):
        for f in os.listdir(DEFECT_SOURCE):
            if f.lower().endswith(valid_exts):
                defect_files.append(os.path.join(DEFECT_SOURCE, f))
    else:
        print(f"Warning: Defect source path not found: {DEFECT_SOURCE}")

    print(f"Found {len(defect_files)} defect images.")

    random.shuffle(defect_files)
    
    for f in defect_files:
        # img = cv2.imread(f)
        # img = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation= cv2.INTER_CUBIC)
        # cv2.imwrite(os.path.join(test_defect_dir, os.path.basename(f)), img)
        
        shutil.copy(f, os.path.join(test_defect_dir, os.path.basename(f)))

    print(f"✅ Dataset organized successfully at: {OUTPUT_ROOT}")
    print(f"   Train Good: {len(train_clean)}")
    print(f"   Test Good:   {len(test_clean)}")
    print(f"   Test Defect: {len(defect_files)}")

if __name__ == "__main__":
    organize_dataset()

# config = {
#     "dataset": {
#         "name": "solar_panel",
#         "root": DATASET_PATH,
#         "normal_dir": "train/good",
#         "abnormal_dir": "test/defect",
#         "task": "classification",
#         "image": {"image_size": [224, 224]},
#     },
#     "dataloader": {
#         "train": {"num_workers": 2}
#     },
# }

# # patchcore_train_infer.py
# from pathlib import Path
# import torch
# from anomalib.data import Folder
# from anomalib.models import Patchcore
# from anomalib.engine import Engine
# from anomalib.deploy import ExportType
# from torchvision.transforms import v2

# from anomalib.engine import Engine


# # --- CONFIGURATION ---
# DATASET_PATH = OUTPUT_ROOT  # Update this to your folder
# SAVE_PATH = "results_patchcore"

# def single_inference(image_path):
#     import torch
#     from anomalib.data.utils import read_image
#     from torchvision.transforms.v2 import Resize, Compose, ToDtype, Normalize
    
#     # 1. Load Model
#     # Note: Anomalib exports a simplified wrapper
#     model = torch.load(f"{SAVE_PATH}/weights/torch/model.pt", weights_only=False)
#     model.eval()

#     # 2. Preprocess Image (Must match training size)
#     image = read_image(image_path)
#     # Convert to tensor and normalize (ImageNet stats are standard)
#     transforms = Compose([
#         Resize((224, 224)),
#         ToDtype(torch.float32, scale=True),
#         Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#     ])
#     input_tensor = transforms(image).unsqueeze(0) # Add batch dimension

#     # 3. Predict
#     with torch.no_grad():
#         output = model(input_tensor)
        
#     # Output contains 'pred_score' (anomaly score) and 'pred_mask' (heatmap)
#     score = output["pred_score"].item()
#     print(f"Anomaly Score: {score:.4f}")
#     if score > 0.5: # Threshold needs tuning based on your validation set
#         print("Defect Detected!")
#     else:
#         print("Clean Panel.")


# transform = v2.Compose([
# v2.Resize((224, 224)),
#     v2.ToDtype(torch.float32, scale=True),
#     v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# ])


# # 'Folder' dataset expects: train/good, test/good, test/defect
# datamodule = Folder(
#     name="solar_panel",
#     root=DATASET_PATH,
#     normal_dir="train/good",      # name of folder inside 'train'
#     abnormal_dir="test/defect",  # name of folder inside 'test'
#     # task="classification",  # or "segmentation" if you have masks
#     # transform=transform,   # ✅ image_size goes here via transforms
    
#     # image_size=(224, 224)   # PatchCore works well at 224
# )
# # datamodule.train_transform = 

# datamodule.setup()
# # engine = Engine(config=config)

# # 2. Initialize Model
# # backbone="wide_resnet50_2" is standard for PatchCore
# # model = Patchcore(backbone="wide_resnet50_2", coreset_sampling_ratio=0.01)
# model = Patchcore(backbone="resnet18", coreset_sampling_ratio=0.01)

# # 3. Train
# engine = Engine(
#     default_root_dir=SAVE_PATH,
#     max_epochs=1,           # PatchCore doesn't really "train", it fits features. 1 epoch is enough.
#     accelerator="cpu",     # Uses GPU if available
#     devices=1,
    
#     enable_progress_bar=False, 
# )

# # engine.fit(model=model)


# print("Training PatchCore...")
# engine.fit(model=model, datamodule=datamodule)



# # 4. Test (Get Accuracy)
# test_results = engine.test(model=model, datamodule=datamodule)
# print(f"Test Accuracy: {test_results}")

# # 5. Export for Inference
# # Exports to OpenVINO or Torch format for production
# engine.export(
#     model=model,
#     export_type=ExportType.TORCH,
#     export_root=Path(SAVE_PATH)
# )
# print(f"Model saved to {SAVE_PATH}")

# single_inference("./dataset/test/defect/sample_crack.jpg")



# import torch
# import random
# import glob
# import os
# import shutil
# from torchvision.transforms import v2
# from anomalib.data import Folder
# from anomalib.models import Patchcore, Padim
# from anomalib.engine import Engine
# from anomalib.deploy import ExportType
# from anomalib import TaskType
# from lightning.pytorch.callbacks import TQDMProgressBar 

# # --- CONFIGURATION ---
# FULL_DATA_ROOT = OUTPUT_ROOT
# SUBSET_DATA_ROOT = "/kaggle/working/solar_dataset_max_coverage"
# EXPORT_DIR = "/kaggle/working/patchcore_model"

# # We aim for 300. If this crashes, lower to 200.
# TARGET_SUBSET_SIZE = 800 

# def train_max_coverage():
#     print("--- 🚀 Configuring Max Coverage Mode ---")
    
#     # 1. SETUP FOLDERS
#     if os.path.exists(SUBSET_DATA_ROOT): shutil.rmtree(SUBSET_DATA_ROOT)
#     os.makedirs(f"{SUBSET_DATA_ROOT}/train/good", exist_ok=True)
#     os.makedirs(f"{SUBSET_DATA_ROOT}/test/good", exist_ok=True)
#     os.makedirs(f"{SUBSET_DATA_ROOT}/test/defect", exist_ok=True)

#     # 2. SMART SAMPLING
#     # We want to spread our selection across the entire 4000 images 
#     # to catch different vendors (assuming files are sorted by date/vendor).
#     all_good = sorted(glob.glob(f"{FULL_DATA_ROOT}/train/good/*"))
    
#     if len(all_good) == 0:
#         print("❌ No images found!")
#         return

#     # Calculate "Step" to pick evenly distributed images
#     # e.g., if you have 4000 images and want 300, pick every 13th image.
#     total_images = len(all_good)
#     step = max(1, total_images // TARGET_SUBSET_SIZE)
    
#     selected_images = all_good[::step] # Python slicing [start:end:step]
    
#     # Trim if we got slightly too many due to rounding
#     selected_images = selected_images[:TARGET_SUBSET_SIZE]
    
#     print(f"Total Available: {total_images}")
#     print(f"Selected: {len(selected_images)} (Picking every {step}th file)")
#     print("This ensures we capture variations from the beginning, middle, and end of your dataset.")

#     # Copy Training Data
#     for p in selected_images:
#         shutil.copy(p, f"{SUBSET_DATA_ROOT}/train/good/")
        
#     # Copy Test Data (Just a few for validation)
#     all_defects = glob.glob(f"{FULL_DATA_ROOT}/test/defect/*")
#     for p in all_defects[:10]: shutil.copy(p, f"{SUBSET_DATA_ROOT}/test/defect/")
#     for p in all_good[:10]: shutil.copy(p, f"{SUBSET_DATA_ROOT}/test/good/")
#         # 
        
#     # 3. LOW RESOLUTION TRANSFORM
    
#     train_transform = v2.Compose([
#            # data alreaydy resized
#         v2.ToImage(), 
#         v2.ToDtype(torch.float32, scale=True),
#         v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#     ])

#     eval_transform = v2.Compose([
        
#         v2.ToImage(), 
#         v2.ToDtype(torch.float32, scale=True),
#         v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#     ])

#     # 4. DATA MODULE
#     datamodule = Folder(
#         name="solar_panel",
#         root=SUBSET_DATA_ROOT, 
#         normal_dir="train/good",      
#         normal_test_dir="test/good",  
#         abnormal_dir="test/defect", 
        
#         num_workers=1,
#         train_batch_size=4,
#         eval_batch_size=4
#     )
#     datamodule.train_transform = train_transform
#     datamodule.eval_transform = eval_transform

#     # 5. MODEL (Still Light)
#     model = Patchcore(
#         backbone="resnet18",
#         pre_trained=True,
#         layers=["layer2"],
#         coreset_sampling_ratio=0.05 # Keep 1% (1% of 300 images is better than 1% of 50)
#     )

#     # 6. TRAIN
#     engine = Engine(
#         max_epochs=1,
#         accelerator="auto",
#         # callbacks=[TQDMProgressBar(refresh_rate=10)],
#     enable_progress_bar=False, 
        
#     )

#     print("Training on Max Coverage Subset...")
#     try:
#         engine.fit(datamodule=datamodule, model=model)
#         print("✅ Success! Trained on diverse subset.")
        
#         results = engine.test(model=model, datamodule=datamodule)
        
#         exported_path = engine.export(
#             model=model,                # Uses the object in RAM (with full bank)
#             export_type="torch", #ExportType.TORCH,
#             export_root=EXPORT_DIR
#         )
#         pt_files = glob.glob(f"{EXPORT_DIR}/**/*.pt", recursive=True)
#         if pt_files:
#             final_path = pt_files[0]
#             print(f"\n✅ FIXED MODEL SAVED AT: {final_path}")
#             print("Use this file in your inference script.")
#         else:
#             print("❌ Export failed.")
            
#         return engine, model, datamodule
#     except Exception as e:
#         print(f"❌ Crashed: {e}")
#         print("Try lowering TARGET_SUBSET_SIZE to 150.")
#         return None, None, None

# if __name__ == "__main__":
#     engine, model, datamodule = train_max_coverage()

import os
os.environ['TRUST_REMOTE_CODE'] = '1'

# from pathlib import Path
# import torch
# import numpy as np
# from anomalib.data.utils import read_image
# from anomalib.deploy import TorchInferencer

# # --- CONFIGURATION ---
# EXPORTED_MODEL_PATH = "/kaggle/working/patchcore_model/weights/torch/model.pt"  # The path returned by engine.export()
# # Example: "/content/exported/model.pt" or the directory

# def single_inference(image_path):
#     """
#     Perform inference on a single image using exported Patchcore model.
    
#     Args:
#         image_path: Path to the image file
    
#     Returns:
#         dict containing predictions
#     """
    
#     # 1. Load the Inferencer (NOT torch.load directly)
#     # The exported model is a TorchScript model that needs Anomalib's Inferencer
#     inferencer = TorchInferencer(
#         path=EXPORTED_MODEL_PATH,  # Path to exported model directory
#         device="cuda" if torch.cuda.is_available() else "cpu"
#     )
    
#     # 2. Perform inference (model handles preprocessing internally)
#     predictions = inferencer.predict(image_path=image_path)
    
#     # 3. Extract results
#     # The output contains multiple keys:
#     # - pred_score: anomaly score (higher = more anomalous)
#     # - pred_label: 0 for normal, 1 for anomaly
#     # - pred_mask: anomaly heatmap (if model was trained with segmentation)
#     # - heat_map: visualization heatmap
    
#     anomaly_score = predictions["pred_score"]
#     is_anomalous = predictions["pred_label"] == 1
#     heatmap = predictions.get("heat_map", None)
    
#     print(f"Image: {Path(image_path).name}")
#     print(f"Anomaly Score: {anomaly_score:.4f}")
#     print(f"Is Anomalous: {is_anomalous}")
    
#     # 4. Get threshold from model metadata (if available)
#     # The model stores the threshold determined during validation
#     threshold = inferencer.model.threshold if hasattr(inferencer.model, 'threshold') else None
#     if threshold:
#         print(f"Model threshold: {threshold:.4f}")
    
#     return predictions

# def batch_inference(image_folder):
#     """
#     Perform inference on a batch of images.
#     """
#     from anomalib.data import Folder
    
#     # Create data module
#     datamodule = Folder(
#         root=image_folder,
#         normal_dir="train/good",  # if your folder has subdirectories
#         abnormal_dir="defect",  # if your folder has subdirectories
#         task="segmentation",  # or "classification" depending on your task
#         image_size=(224, 224)
#     )
    
#     # Create inferencer
#     inferencer = TorchInferencer(
#         path=EXPORTED_MODEL_PATH,
#         device="cuda" if torch.cuda.is_available() else "cpu"
#     )
    
#     # Get predictions for all images
#     predictions = []
#     for batch in datamodule.val_dataloader():
#         batch_predictions = inferencer.predict(batch)
#         predictions.extend(batch_predictions)
    
#     return predictions

# # Example usage:
# if __name__ == "__main__":
#     # Single image inference
#     image_path = "/kaggle/input/test-anomaly-cracks/crack/000014_jpg.rf.ce1dabe2f2037486d170a4cd1c0516be_cell_127.jpg"
#     result = single_inference(image_path)
    
#     # If you want to visualize the results:
#     import matplotlib.pyplot as plt
    
#     # Show original image with heatmap overlay
#     if "heat_map" in result:
#         plt.figure(figsize=(12, 4))
        
#         plt.subplot(1, 3, 1)
#         plt.imshow(result["image"])
#         plt.title("Original Image")
        
#         plt.subplot(1, 3, 2)
#         plt.imshow(result["heat_map"], cmap='hot')
#         plt.title("Anomaly Heatmap")
#         plt.colorbar()
        
#         plt.subplot(1, 3, 3)
#         # Overlay heatmap on image
#         plt.imshow(result["image"])
#         plt.imshow(result["heat_map"], cmap='hot', alpha=0.5)
#         plt.title(f"Overlay (Score: {result['pred_score']:.3f})")
        
#         plt.tight_layout()
#         plt.show()

# from anomalib.deploy import TorchInferencer
# from anomalib.data.utils import read_image

# def get_anomaly_score(image_path, model_path):
#     inferencer = TorchInferencer(path=model_path, device="cpu")
#     image = read_image(image_path)
#     predictions = inferencer.predict(image)
    
#     # Extract score safely
#     score_tensor = predictions['pred_score']
#     score = score_tensor.item() if hasattr(score_tensor, 'item') else float(score_tensor)
    
#     # Extract label safely
#     label_tensor = predictions['pred_label']
#     label = label_tensor.item() if hasattr(label_tensor, 'item') else int(label_tensor)
    
#     return score, label

# # Usage
# score, label = get_anomaly_score(
#     # "/kaggle/input/test-anomaly-cracks/crack/000014_jpg.rf.ce1dabe2f2037486d170a4cd1c0516be_cell_127.jpg",
#     "/kaggle/working/pv_organized_dataset_crack/test/good/DSC_0496_cell18.jpg",
#     "/kaggle/working/patchcore_model/weights/torch/model.pt"
# )

# print(f"Score: {score:.4f}")
# print(f"Is anomaly: {label == 1}")

"""
Improved Preprocessing Pipeline for EL PV Cell Anomaly Detection
Fixes: Aspect ratio preservation, brightness normalization, consistent pipeline
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, Optional
import albumentations as A
from albumentations.pytorch import ToTensorV2


class RobustPreprocessor:
    """
    Production-ready preprocessing pipeline with aspect ratio preservation
    and brightness normalization
    """

    def __init__(self, target_size: int = 320, use_clahe: bool = True):
        self.target_size = target_size
        self.use_clahe = use_clahe

        # Initialize CLAHE for brightness normalization
        if use_clahe:
            self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    def validate_image(self, img: np.ndarray) -> Tuple[bool, str]:
        """
        Validate image quality before processing

        Returns:
            (is_valid, error_message)
        """
        if img is None:
            return False, "Failed to load image"

        if len(img.shape) < 2:
            return False, "Invalid image dimensions"

        # Convert to grayscale if needed
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Check brightness
        mean_brightness = np.mean(img)
        if mean_brightness < 10:
            return False, "Image too dark (mean brightness < 10)"
        if mean_brightness > 245:
            return False, "Image overexposed (mean brightness > 245)"

        # Check contrast
        std_brightness = np.std(img)
        if std_brightness < 5:
            return False, "Insufficient contrast (std < 5)"

        return True, "OK"

    def resize_with_padding(self, img: np.ndarray,
                            target_size: Optional[int] = None) -> np.ndarray:
        """
        Resize image while maintaining aspect ratio using padding

        Args:
            img: Input image (grayscale or color)
            target_size: Target size (default: self.target_size)

        Returns:
            Padded square image
        """
        if target_size is None:
            target_size = self.target_size

        h, w = img.shape[:2]

        # Calculate scaling factor
        scale = target_size / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)

        # Resize while maintaining aspect ratio
        resized = cv2.resize(img, (new_w, new_h),
                             interpolation=cv2.INTER_CUBIC)

        # Create canvas with padding
        if len(img.shape) == 3:
            canvas = np.zeros(
                (target_size, target_size, img.shape[2]), dtype=np.uint8)
        else:
            canvas = np.zeros((target_size, target_size), dtype=np.uint8)

        # Center the image
        y_offset = (target_size - new_h) // 2
        x_offset = (target_size - new_w) // 2
        canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized

        return canvas

    def resize_with_padding(self, img, target_size=(320, 320)):
        """
        يقوم بقراءة الصورة، تحويلها لمربع باستخدام BORDER_REPLICATE، 
        ثم تغيير حجمها للمقاس المطلوب دون تشويه.
        """
        # 1. قراءة الصورة
        # img = cv2.imread(image_path)
        if img is None:
            print(f"لم يتم العثور على الصورة: {image_path}")
            return None
    
        # 2. الحصول على أبعاد الصورة الحالية
        h, w = img.shape[:2]
    
        # 3. حساب الحواف المطلوبة لجعل الصورة مربعة
        if h > w:
            # الصورة أطول من عرضها، نضيف حواف يميناً ويساراً
            pad_top = 0
            pad_bottom = 0
            pad_left = (h - w) // 2
            pad_right = h - w - pad_left
        elif w > h:
            # الصورة أعرض من طولها، نضيف حواف أعلى وأسفل
            pad_top = (w - h) // 2
            pad_bottom = w - h - pad_top
            pad_left = 0
            pad_right = 0
        else:
            # الصورة مربعة بالفعل
            pad_top, pad_bottom, pad_left, pad_right = 0, 0, 0, 0
    
        # 4. إضافة الحواف باستخدام BORDER_REPLICATE
        # هذا الأمر سيقوم بنسخ آخر بيكسلات في أطراف الصورة لتعبئة الفراغ
        padded_img = cv2.copyMakeBorder(
            img, 
            pad_top, pad_bottom, pad_left, pad_right, 
            cv2.BORDER_REPLICATE
        )
    
        # 5. تغيير الحجم للمقاس النهائي المطلوب (مثلاً 224x224)
        final_img = cv2.resize(padded_img, target_size)
    
        return final_img


    def apply_clahe(self, img: np.ndarray) -> np.ndarray:
        """Apply Contrast Limited Adaptive Histogram Equalization"""
        if len(img.shape) == 3:
            # Convert to grayscale if color
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        return self.clahe.apply(img)

    def preprocess_for_training(self, image_path: str) -> Tuple[np.ndarray, dict]:
        """
        Preprocess image for training

        Returns:
            (preprocessed_image, metadata)
        """
        # Load image
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        
        # Validate
        is_valid, message = self.validate_image(img)
        # if not is_valid:
            # print("not valid", image_path)
            # raise ValueError(f"Invalid image {image_path}: {message}")

        metadata = {
            'original_shape': img.shape,
            'mean_brightness': float(np.mean(img)),
            'std_brightness': float(np.std(img)),
            'is_valid': is_valid
        }

        # Apply CLAHE for brightness normalization
        if self.use_clahe:
            img = self.apply_clahe(img)
        
        # Resize with padding to maintain aspect ratio
        img = self.resize_with_padding(img)

        metadata['preprocessed_shape'] = img.shape

        return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB), metadata

    def preprocess_for_inference(self, image_path: str) -> Tuple[np.ndarray, dict]:
        """
        Preprocess image for inference (identical to training preprocessing)

        Returns:
            (preprocessed_image, metadata)
        """
        # Use the same pipeline as training
        return self.preprocess_for_training(image_path)


# def get_training_augmentation(img_size: int = 320) -> A.Compose:
#     """
#     Get augmentation pipeline for training
#     Realistic augmentations suitable for EL images
#     """
#     return A.Compose([
#         # Already resized with padding, so no resize here

#         # Geometric augmentations
#         A.HorizontalFlip(p=0.5),
#         A.VerticalFlip(p=0.5),
#         A.Rotate(limit=5, p=0.5, border_mode=cv2.BORDER_CONSTANT, value=0),
#         A.ShiftScaleRotate(
#             shift_limit=0.05,
#             scale_limit=0.05,
#             rotate_limit=5,
#             p=0.3,
#             border_mode=cv2.BORDER_CONSTANT,
#             value=0
#         ),

#         # Intensity augmentations (crucial for EL images)
#         A.RandomBrightnessContrast(
#             brightness_limit=0.15,
#             contrast_limit=0.15,
#             p=0.5
#         ),
#         A.GaussNoise(var_limit=(5.0, 20.0), p=0.2),
#         A.GaussianBlur(blur_limit=(3, 5), p=0.1),

#         # Convert to tensor and normalize
#         # Note: Will compute dataset-specific statistics
#         A.Normalize(mean=[0.5], std=[0.5]),  # Placeholder, should be computed
#         ToTensorV2(),
#     ])


# def get_inference_augmentation(img_size: int = 320) -> A.Compose:
#     """
#     Get augmentation pipeline for inference (no augmentation, just normalization)
#     """
#     return A.Compose([
#         # Already resized with padding
#         # Just normalize
#         A.Normalize(mean=[0.5], std=[0.5]),  # Should match training stats
#         ToTensorV2(),
#     ])


def compute_dataset_normalization_stats(image_paths: list,
                                        preprocessor: RobustPreprocessor) -> Tuple[float, float]:
    """
    Compute dataset-specific normalization statistics
    
    Args:
        image_paths: List of image paths
        preprocessor: Preprocessor instance
    
    Returns:
        (mean, std) for the dataset
    """
    print("Computing dataset normalization statistics...")
    
    pixel_values = []
    
    for img_path in image_paths[:1000]:  # Sample 1000 images
        try:
            img, _ = preprocessor.preprocess_for_training(img_path)
            pixel_values.extend(img.flatten().tolist())
        except Exception as e:
            print(f"Skipping {img_path}: {e}")
            continue
    
    pixel_values = np.array(pixel_values) / 255.0  # Normalize to [0, 1]
    
    mean = float(np.mean(pixel_values))
    std = float(np.std(pixel_values))
    
    print(f"Dataset statistics: mean={mean:.4f}, std={std:.4f}")
    
    return mean, std


# # Example usage
# if __name__ == "__main__":
#     # Initialize preprocessor
#     preprocessor = RobustPreprocessor(target_size=320, use_clahe=True)

#     # Test preprocessing
#     test_image = "path/to/test/image.jpg"

#     try:
#         # For training
#         img_train, metadata = preprocessor.preprocess_for_training(test_image)
#         print(f"Training preprocessing: {metadata}")

#         # For inference (same pipeline)
#         img_infer, metadata = preprocessor.preprocess_for_inference(test_image)
#         print(f"Inference preprocessing: {metadata}")

#         # Verify they're identical
#         assert img_train.shape == img_infer.shape
#         print("✅ Training and inference preprocessing are consistent!")

#     except Exception as e:
#         print(f"❌ Error: {e}")


# Train Good: 4318
#    Test Good:   1851
#    Test Defect: 1264

import torch
import random
import glob
import os
import shutil
from torchvision.transforms import v2
from anomalib.data import Folder
from anomalib.models import Patchcore
from anomalib.engine import Engine
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
import numpy as np

# --- CONFIGURATION ---
FULL_DATA_ROOT = OUTPUT_ROOT
SUBSET_DATA_ROOT = "/kaggle/working/solar_dataset_robust"
EXPORT_DIR = "/kaggle/working/patchcore_model_robust_2"

# TRAINING SETTINGS
TARGET_TRAIN_SIZE = 1000  # Good for industrial inspection
TARGET_VAL_SIZE = 1000    # Validation set (also normal images)
TARGET_TEST_SIZE = 1000   # Test images (50 normal + 50 defect)

def create_max_coverage_dataset():
    """Create dataset with train/val/test splits."""
    print("--- 🚀 Creating Max Coverage Dataset ---")
    
    # Clean and create directories
    if os.path.exists(SUBSET_DATA_ROOT):
        shutil.rmtree(SUBSET_DATA_ROOT)
    
    # Create all necessary directories
    for split in ['train', 'val', 'test']:
        for category in ['good', 'defect']:
            os.makedirs(f"{SUBSET_DATA_ROOT}/{split}/{category}", exist_ok=True)
    
    # Get all normal images
    all_normal = sorted(glob.glob(f"{FULL_DATA_ROOT}/train/good/*"))
    all_defect = sorted(glob.glob(f"{FULL_DATA_ROOT}/test/defect/*"))
    
    if len(all_normal) == 0 or len(all_defect) == 0:
        print("❌ No images found!")
        return False
    
    print(f"Found {len(all_normal)} normal images, {len(all_defect)} defect images")
    
    # 1. STRATIFIED SAMPLING (even distribution)
    total_normal = len(all_normal)
    
    # Calculate indices for train/val split
    step = max(1, total_normal // (TARGET_TRAIN_SIZE + TARGET_VAL_SIZE))
    all_normal_sampled = all_normal[::step]
    
    # Split into train and validation
    train_images = all_normal_sampled[:TARGET_TRAIN_SIZE]
    val_images = all_normal_sampled[TARGET_TRAIN_SIZE:TARGET_TRAIN_SIZE + TARGET_VAL_SIZE]
    
    # 2. TEST SET (balanced: normal + defect)
    # Get normal images not used in train/val
    remaining_normal = [img for img in all_normal if img not in train_images and img not in val_images]
    step_normal_test = max(1, len(remaining_normal) // (TARGET_TEST_SIZE//2))
    test_normal = remaining_normal[::step_normal_test][:TARGET_TEST_SIZE//2]
    
    # Sample defect images for test
    step_defect = max(1, len(all_defect) // (TARGET_TEST_SIZE//2))
    test_defect = all_defect[::step_defect][:TARGET_TEST_SIZE//2]
    
    # Copy files
    print(f"\n📊 Dataset Statistics:")
    print(f"  Training: {len(train_images)} normal")
    print(f"  Validation: {len(val_images)} normal")
    print(f"  Test: {len(test_normal)} normal + {len(test_defect)} defect")
    
    for img in train_images:
        shutil.copy(img, f"{SUBSET_DATA_ROOT}/train/good/")
    for img in val_images:
        shutil.copy(img, f"{SUBSET_DATA_ROOT}/val/good/")
    for img in test_normal:
        shutil.copy(img, f"{SUBSET_DATA_ROOT}/test/good/")
    for img in test_defect:
        shutil.copy(img, f"{SUBSET_DATA_ROOT}/test/defect/")
    
    print("✅ Dataset created successfully!")
    return True

def create_folder_structure_for_anomalib():
    """
    Anomalib's Folder class expects specific structure.
    We need to combine train and val into one directory with subfolders.
    """
    print("\n--- 📁 Creating Anomalib-compatible folder structure ---")
    
    # Create MVTec-like structure that Anomalib expects
    ANOMALIB_ROOT = "/kaggle/working/anomalib_dataset"
    
    if os.path.exists(ANOMALIB_ROOT):
        shutil.rmtree(ANOMALIB_ROOT)
    
    # Structure: root/train/good/*.jpg
    #           root/test/good/*.jpg
    #           root/test/defect/*.jpg
    os.makedirs(f"{ANOMALIB_ROOT}/train/good", exist_ok=True)
    os.makedirs(f"{ANOMALIB_ROOT}/test/good", exist_ok=True)
    os.makedirs(f"{ANOMALIB_ROOT}/test/defect", exist_ok=True)
    
    preprocessor = RobustPreprocessor(target_size=320, use_clahe=False)
    # Copy training images (from both train and val - we'll use val_split_ratio)
    train_images = glob.glob(f"{SUBSET_DATA_ROOT}/train/good/*") + glob.glob(f"{SUBSET_DATA_ROOT}/val/good/*")

    train_count = 0
    for i, img_path in enumerate(train_images):
        img, metadata = preprocessor.preprocess_for_training(img_path)
        if metadata['is_valid']:
            train_count+=1
            cv2.imwrite(f"{ANOMALIB_ROOT}/train/good/img_{i:04d}.jpg", img)
        # shutil.copy(img_path, f"{ANOMALIB_ROOT}/train/good/img_{i:04d}.jpg")
    
    # Copy test images
    test_normal = glob.glob(f"{SUBSET_DATA_ROOT}/test/good/*")
    test_normal_count = 0
    for i, img_path in enumerate(test_normal):
        img, metadata = preprocessor.preprocess_for_inference(img_path)
        if metadata['is_valid']:
            test_normal_count += 1
            cv2.imwrite(f"{ANOMALIB_ROOT}/test/good/img_{i:04d}.jpg", img)
        
        # shutil.copy(img_path, f"{ANOMALIB_ROOT}/test/good/img_{i:04d}.jpg")
    
    test_defect = glob.glob(f"{SUBSET_DATA_ROOT}/test/defect/*")
    test_defect_count = 0
    for i, img_path in enumerate(test_defect):
        img, metadata = preprocessor.preprocess_for_inference(img_path)
        if metadata['is_valid']:
            test_defect_count+=1
            cv2.imwrite(f"{ANOMALIB_ROOT}/test/defect/img_{i:04d}.jpg", img)
        # shutil.copy(img_path, f"{ANOMALIB_ROOT}/test/defect/img_{i:04d}.jpg")
    
    print(f"Created Anomalib dataset with:")
    print(f"  Train: {len(train_images)} normal images but got {train_count}")
    print(f"  Test: {len(test_normal)} normal but got {test_normal_count} + {len(test_defect)} defect images but got {test_defect_count}")
    
    return ANOMALIB_ROOT

# def train_robust_patchcore():
"""Train robust Patchcore model."""

create_max_coverage_dataset()


# Create Anomalib-compatible folder structure
DATASET_ROOT = create_folder_structure_for_anomalib()


# compute_dataset_normalization_stats([os.path.join(DATASET_ROOT+"/train/good", img_path) for img_path in os.listdir(DATASET_ROOT+"/train/good")], RobustPreprocessor())

# Train:  1285
# Test: 132 normal + 149 defect images

import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import glob

def compute_dataset_statistics(dataset_path):
    """
    Compute mean and std for your specific dataset.
    
    Args:
        dataset_path: Path to your training images (normal images only)
    
    Returns:
        mean, std: Computed statistics for your dataset
    """
    # Get all image paths
    image_paths = glob.glob(f"{dataset_path}/**/*.jpg") + \
                  glob.glob(f"{dataset_path}/**/*.png") + \
                  glob.glob(f"{dataset_path}/**/*.jpeg")
    
    print(f"Found {len(image_paths)} images for statistics computation")
    
    # Initialize accumulators
    mean = torch.zeros(3)
    std = torch.zeros(3)
    pixel_count = 0
    
    transform = transforms.ToTensor()
    
    for i, img_path in enumerate(image_paths):
        try:
            # Load image
            img = Image.open(img_path).convert('RGB')
            
            # Convert to tensor
            img_tensor = transform(img)
            
            # Accumulate mean
            mean += img_tensor.mean(dim=(1, 2))
            
            # Accumulate std
            std += img_tensor.std(dim=(1, 2))
            
            pixel_count += 1
            
            if i % 100 == 0:
                print(f"Processed {i}/{len(image_paths)} images...")
                
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
    
    # Compute averages
    mean /= pixel_count
    std /= pixel_count
    
    print(f"\n📊 Computed Statistics for {pixel_count} images:")
    print(f"Mean: {mean.tolist()}")
    print(f"Std: {std.tolist()}")
    
    return mean.tolist(), std.tolist()

# Usage in your training setup
def get_custom_normalization():
    """Compute and use dataset-specific normalization."""
    
    # Compute statistics on your training data
    dataset_path = f"{SUBSET_DATA_ROOT}/train/good"  # Use only normal images
    mean, std = compute_dataset_statistics(dataset_path)
    
    # Alternative: If computation takes too long, use ImageNet or default values
    # mean = [0.5, 0.5, 0.5]
    # std = [0.5, 0.5, 0.5]
    
    return transforms.Normalize(mean=mean, std=std)

try:
    # Get all training images
    train_images = glob.glob(f"{DATASET_ROOT}/train/good/*.jpg")
    
    # Compute statistics
    mean, std = compute_dataset_statistics(f"{DATASET_ROOT}/train")
    
    print(f"✅ Using computed statistics:")
    print(f"   Mean: {mean}")
    print(f"   Std: {std}")
except Exception as e:
    
    print("⚠️ Could not compute statistics, using defaults...", str(e))
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

# Mean: [0.24394622445106506, 0.24394622445106506, 0.24394622445106506]
   # Std: [0.2631329298019409, 0.2631329298019409, 0.2631329298019409]

# 1. DATA AUGMENTATION FOR ROBUSTNESS
# get_training_augmentation()
# v2.Compose([
#     # v2.Resize((256, 256)),  # Uncomment if images vary in size
#     v2.RandomHorizontalFlip(p=0.5),
#     v2.RandomVerticalFlip(p=0.5),
#     v2.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
#     v2.ToImage(),
#     v2.ToDtype(torch.float32, scale=True),
#     v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# ])
train_transform = v2.Compose([
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandomVerticalFlip(p=0.5),
        # v2.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=mean, std=std)  # Use YOUR computed statistics
    ])

# No augmentation for evaluation
eval_transform = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=mean, std=std)  # Use same statistics
    ])
# get_inference_augmentation()
# v2.Compose([
#     # v2.Resize((256, 256)),
#     v2.ToImage(),
#     v2.ToDtype(torch.float32, scale=True),
#     v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# ])

# 2. DATA MODULE - CORRECT WAY
datamodule = Folder(
    name="solar_panel_robust",
    root=DATASET_ROOT,
    normal_dir="train/good",      # All normal images for training
    normal_test_dir="test/good",  # Normal images for testing
    abnormal_dir="test/defect",   # Defect images for testing
    
    # Batch sizes
    train_batch_size=16,
    eval_batch_size=16,
    
    # Workers
    num_workers=4,
    
    # VALIDATION: Split training data automatically
    val_split_mode="from_test",  # Split from training data
    val_split_ratio=0.5,  # 20% of training data for validation
    
    # Random seed for reproducibility
    seed=42,
)

datamodule.train_transform = train_transform
datamodule.eval_transform = eval_transform

# 3. ROBUST PATCHCORE CONFIGURATION
model = Patchcore(
    # Backbone
    # backbone="wide_resnet50_2",  # Better feature extraction
    # layers=["layer2", "layer3"],  # Multi-scale features
    
    # backbone="resnet34",  # Better feature extraction
    backbone="resnet18",  # Better feature extractio
    layers=["layer2", "layer3"],  # Multi-scale features

    pre_trained=True,
    
    # Coreset configuration
    coreset_sampling_ratio=0.01,  # 1% sampling
    num_neighbors=9,  # More neighbors for stability
)


# 5. ENGINE CONFIGURATION
engine = Engine(
    max_epochs=1,
    accelerator="auto",
    devices=1,
    # callbacks=callbacks,
    enable_progress_bar=False,
    # Logging
    default_root_dir=EXPORT_DIR,
)

print("\n--- 🚀 Training Robust Patchcore ---")
print(f"Dataset root: {DATASET_ROOT}")

try:
    # Train the model
    engine.fit(
        model=model,
        datamodule=datamodule,
    )
    
    print("\n✅ Training completed!")
    
    # Load best model and test
    print("\n--- 📊 Testing Best Model ---")
    
    # Test on test set
    test_results = engine.test(
        model=model,
        datamodule=datamodule,
        # ckpt_path=best_checkpoint
    )
    
    # Export the model
    print("\n--- 💾 Exporting Model ---")
    exported_path = engine.export(
        model=model,
        export_type="torch",
        export_root=EXPORT_DIR
    )
    
    print(f"✅ Model exported to: {exported_path}")
    
    # return engine, model, datamodule
    
except Exception as e:
    print(f"❌ Training failed: {e}")
    import traceback
    traceback.print_exc()
    # return None


# if __name__ == "__main__":
#     # Choose your training method:
    
#     # Method 1: Robust training with validation (recommended)
#     print("=" * 60)
#     print("METHOD 1: Robust Training with Validation")
#     print("=" * 60)
#     engine, model, datamodule = train_robust_patchcore()


predictions = engine.predict(model, datamodule=datamodule)

y_true = []
y_scores = []

for batch in predictions:
    y_true.extend(batch["gt_label"].cpu().numpy())
    # y_scores.extend(batch["anomaly_map"].cpu().numpy())
    batch_scores = batch["anomaly_map"].mean(dim=[1,2]).cpu().numpy()
    y_scores.extend(batch_scores)

anomaly_map = batch["anomaly_map"].cpu().numpy()[0,0]
score = np.percentile(anomaly_map, 95)
score

from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

fpr, tpr, thresholds = roc_curve(y_true, y_scores)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.3f})")
plt.plot([0, 1], [0, 1], linestyle="--")  # random line
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()



optimal_idx = np.argmax(tpr - fpr)
optimal_threshold = thresholds[optimal_idx]

print("Best threshold:", optimal_threshold)


np.percentile(y_scores, 98)

from sklearn.metrics import confusion_matrix

y_pred = (np.array(y_scores) > optimal_threshold).astype(int)
cm = confusion_matrix(y_true, y_pred)
print(cm)


from sklearn.metrics import roc_auc_score, f1_score, roc_curve
import numpy as np

# AUROC
auroc = roc_auc_score(y_true, y_scores)
print("AUROC:", auroc)

# أفضل threshold بناءً على F1
fpr, tpr, thresholds = roc_curve(y_true, y_scores)

best_f1 = 0
best_threshold = 0
for th in thresholds:
    y_pred = (np.array(y_scores) > th).astype(int)
    f1 = f1_score(y_true, y_pred)
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = th

print("Best threshold:", best_threshold)
print("Best F1:", best_f1)


import numpy as np
import matplotlib.pyplot as plt

normal_scores = np.array(y_scores)[np.array(y_true)==0]
defect_scores = np.array(y_scores)[np.array(y_true)==1]

print("Normal mean:", normal_scores.mean())
print("Defect mean:", defect_scores.mean())

print("Normal max:", normal_scores.max())
print("Defect min:", defect_scores.min())



plt.hist(normal_scores, bins=30, alpha=0.5, label='Normal')
plt.hist(defect_scores, bins=30, alpha=0.5, label='Defect')
plt.axvline(best_threshold, color='red', linestyle='--', label='Best Threshold')
plt.legend()
plt.xlabel('Anomaly Score')
plt.ylabel('Count')
plt.title('Distribution of Anomaly Scores')
plt.show()


# import torch
# import json
# import os

# # 1. Path to save
# EXPORT_DIR = "/kaggle/working/patchcore_model_robust_final_auc_99_f1_98"
# os.makedirs(EXPORT_DIR, exist_ok=True)

# # 2. Save the PyTorch model weights
# model_path = os.path.join(EXPORT_DIR, "patchcore_model.pt")
# torch.save(model.state_dict(), model_path)

# # 3. Save the best threshold
# threshold_path = os.path.join(EXPORT_DIR, "best_threshold.json")
# with open(threshold_path, "w") as f:
#     json.dump({"best_threshold": float(best_threshold)}, f)

# print(f"✅ Model saved at: {model_path}")
# print(f"✅ Threshold saved at: {threshold_path}")


import zipfile
import os
from IPython.display import FileLink

# Assuming your folder is 'my_results_folder' in the working directory
zip_file_name = 'patchcore_model_robust_final_auc_99_f1_99_corset05_res18_diffaug_diffresize_noclahe_export.zip' 
folder_to_zip = '/kaggle/working/patchcore_model_robust/weights/torch' # Or '/kaggle/working/my_results_folder'
folder_to_zip = EXPORT_DIR+'/weights/torch'
# Create the zip file
with zipfile.ZipFile(zip_file_name, 'w', zipfile.ZIP_DEFLATED) as zipf:
    for root, dirs, files in os.walk(folder_to_zip):
        for file in files:
            if ".pt" in file or '.ckpt' in file:
                zipf.write(os.path.join(root, file), os.path.relpath(os.path.join(root, file), folder_to_zip))

# Display download link (optional but helpful)
display(FileLink(zip_file_name))


EXPORT_DIR

os.listdir(EXPORT_DIR+"/Patchcore/solar_panel_robust/latest/weights/lightning")
os.listdir(EXPORT_DIR+"/Patchcore/solar_panel_robust/latest")



import os
os.environ['TRUST_REMOTE_CODE'] = '1'

def infer_single_image(image_path, model_path=None):
    """
    Simple inference for a single image.
    """
    import os
    os.environ['TRUST_REMOTE_CODE'] = '1'
    
    from anomalib.deploy import TorchInferencer
    from anomalib.data.utils import read_image
    
    if model_path is None:
        # Find the latest model
        exported_files = []
        for root, dirs, files in os.walk(EXPORT_DIR):
            for file in files:
                if file.endswith('.pt'):
                    exported_files.append(os.path.join(root, file))
        if exported_files:
            model_path = exported_files[0]
        else:
            print("❌ No model found!")
            return None
    
    # Create inferencer
    inferencer = TorchInferencer(
        path=model_path,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    # Predict
    image = read_image(image_path)
    predictions = inferencer.predict(image)
    
    # Extract results
    score_tensor = predictions['pred_score']
    score = score_tensor.item() if hasattr(score_tensor, 'item') else float(score_tensor)
    
    label_tensor = predictions['pred_label']
    label = label_tensor.item() if hasattr(label_tensor, 'item') else int(label_tensor)
    
    print(f"\nResults for: {os.path.basename(image_path)}")
    print(f"Anomaly Score: {score:.4f}")
    print(f"Prediction: {'DEFECT' if label == 1 else 'NORMAL'}")
    
    return {
        'score': score,
        'label': label,
        'prediction': 'DEFECT' if label == 1 else 'NORMAL',
        # 'heatmap': predictions.get('heat_map', None)
    }

infer_single_image(
    # "/kaggle/input/test-anomaly-cracks/crack/000014_jpg.rf.ce1dabe2f2037486d170a4cd1c0516be_cell_127.jpg",
    "/kaggle/working/pv_organized_dataset_crack/test/good/DSC_0496_cell18.jpg",
    "/kaggle/working/patchcore_model_robust/weights/torch/model.pt")

# def batch_inference(image_folder):
#     """
#     Perform inference on a batch of images.
#     """
#     from anomalib.data import Folder
    
#     # Create data module
#     datamodule = Folder(
#         root=image_folder,
#         normal_dir="train/good",  # if your folder has subdirectories
#         abnormal_dir="defect",  # if your folder has subdirectories
#         task="segmentation",  # or "classification" depending on your task
#         image_size=(224, 224)
#     )
    
#     # Create inferencer
#     inferencer = TorchInferencer(
#         path=EXPORTED_MODEL_PATH,
#         device="cuda" if torch.cuda.is_available() else "cpu"
#     )
    
#     # Get predictions for all images
#     predictions = []
#     for batch in datamodule.val_dataloader():
#         batch_predictions = inferencer.predict(batch)
#         predictions.extend(batch_predictions)
    
#     return predictions
    

MODEL_PATH = "/kaggle/working/patchcore_model_robust/weights/torch/model.pt"

def analyze_score_distribution(model_path, good_dir, defect_dir):
    """Analyze score distribution to understand what's normal."""
    import glob
    import numpy as np
    import matplotlib.pyplot as plt
    
    inferencer = TorchInferencer(path=model_path, device="cpu")
    
    # Collect scores
    good_scores = []
    defect_scores = []
    
    # Good images
    for img_path in glob.glob(f"{good_dir}/*.jpg")[:50]:
        image = read_image(img_path)
        predictions = inferencer.predict(image)
        good_scores.append(predictions['pred_score'].item())
    
    # Defect images
    for img_path in glob.glob(f"{defect_dir}/*.jpg")[:50]:
        image = read_image(img_path)
        predictions = inferencer.predict(image)
        defect_scores.append(predictions['pred_score'].item())
    
    # Statistics
    print("Good images score stats:")
    print(f"  Min: {np.min(good_scores):.4f}")
    print(f"  Max: {np.max(good_scores):.4f}")
    print(f"  Mean: {np.mean(good_scores):.4f}")
    print(f"  Std: {np.std(good_scores):.4f}")
    
    print("\nDefect images score stats:")
    print(f"  Min: {np.min(defect_scores):.4f}")
    print(f"  Max: {np.max(defect_scores):.4f}")
    print(f"  Mean: {np.mean(defect_scores):.4f}")
    print(f"  Std: {np.std(defect_scores):.4f}")
    
    # Visualize
    plt.figure(figsize=(10, 6))
    plt.hist(good_scores, alpha=0.5, label='Good', bins=20)
    plt.hist(defect_scores, alpha=0.5, label='Defect', bins=20)
    plt.xlabel('Anomaly Score')
    plt.ylabel('Frequency')
    plt.title('Score Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    # Suggest threshold
    overlap_threshold = (np.mean(good_scores) + np.mean(defect_scores)) / 2
    print(f"\nSuggested threshold: {overlap_threshold:.4f}")
    
    return good_scores, defect_scores, overlap_threshold

# Run analysis
good_scores, defect_scores, threshold = analyze_score_distribution(
    MODEL_PATH,
    "/kaggle/working/pv_organized_dataset_crack/test/good",
    # "/kaggle/working/pv_organized_dataset_crack/test/defect"
    "/kaggle/input/test-microcrack/all_micro_crack"
)

from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def plot_confusion_matrix(good_scores, defect_scores, threshold):
    print(f"\n--- Generating Confusion Matrix (Threshold: {threshold:.4f}) ---")
    
    # 1. Create Ground Truth Labels
    # 0 = Good, 1 = Defect
    y_true = ([0] * len(good_scores)) + ([1] * len(defect_scores))
    
    # 2. Create Predictions
    # If score > threshold, we predict Defect (1). Otherwise Good (0).
    # Note: We assume higher score = more anomalous
    all_scores = good_scores + defect_scores
    y_pred = [1 if score > threshold else 0 for score in all_scores]
    
    # 3. Generate Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # 4. Visualize it
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Predicted Good', 'Predicted Defect'],
                yticklabels=['Actual Good', 'Actual Defect'])
    
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()
    
    # 5. Print Detailed Metrics
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=['Good', 'Defect']))

# === EXECUTION ===
# 1. Calculate the threshold (using the logic from your previous script)
# Or set it manually if you prefer, e.g., threshold = 0.90
# threshold = (np.mean(good_scores) + np.mean(defect_scores)) / 2

# 2. Plot
plot_confusion_matrix(good_scores, defect_scores, threshold)

import matplotlib.pyplot as plt
import glob
import cv2
import numpy as np
import os
import random
import torch
from anomalib.deploy import TorchInferencer
from anomalib.data.utils import read_image

# --- CONFIGURATION ---
# THRESHOLD = 0.8 

def verify_with_heatmap_overlay(model_path, image_dir):
    print("\n--- 🔍 Generating Heatmap Overlays ---")

    inferencer = TorchInferencer(path=model_path, device="cpu")
    
    # Get image paths
    image_paths = glob.glob(os.path.join(image_dir, "*.jpg")) + \
                  glob.glob(os.path.join(image_dir, "*.png"))
    
    if not image_paths:
        print("No images found!")
        return

    # Pick 5 random images
    selected_paths = random.sample(image_paths, min(len(image_paths), 5))

    fig, axes = plt.subplots(1, len(selected_paths), figsize=(20, 5))
    if len(selected_paths) == 1: axes = [axes] 

    for i, img_path in enumerate(selected_paths):
        # 1. Inference
        image = read_image(img_path) 
        result = inferencer.predict(image)
        
        # 2. Extract Data
        score = result.pred_score
        # FIX 1: Ensure anomaly map is on CPU and detached from graph
        anomaly_map = result.anomaly_map.squeeze().cpu()
        
        # 3. Normalize Heatmap
        if anomaly_map.max() - anomaly_map.min() == 0:
            amap_norm = anomaly_map
        else:
            amap_norm = (anomaly_map - anomaly_map.min()) / (anomaly_map.max() - anomaly_map.min())
        
        # FIX 2: Convert to Numpy BEFORE using .astype()
        amap_numpy = amap_norm.numpy()
        amap_uint8 = (amap_numpy * 255).astype(np.uint8)

        # anomaly_map = result.anomaly_map.squeeze().cpu().numpy()

        # 2. FIXED NORMALIZATION
        # Don't use the image's own max. Use a "Global Max" threshold.
        # Any score above this value is "100% Defect" (Red).
        # You can tune this. Start with 1.0 or the max value from your histogram (0.14).
        # GLOBAL_MAX_SCORE = 0.8  # Set this slightly higher than your histogram's defect peak
        
        # Clip values to the global max to prevent outliers from distorting scale
        # anomaly_map_clipped = np.clip(anomaly_map, 0, GLOBAL_MAX_SCORE)
        
        # Normalize based on the GLOBAL range
        # amap_norm = anomaly_map_clipped / GLOBAL_MAX_SCORE 
        
        # 3. Convert to Color (Same as before)
        # amap_uint8 = (amap_norm * 255).astype(np.uint8)
        
        
        # 4. Colorize
        # Setup original image
        orig_img = cv2.imread(img_path)
        orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
        
        # Resize heatmap to match original image size
        heatmap = cv2.applyColorMap(amap_uint8, cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        heatmap = cv2.resize(heatmap, (orig_img.shape[1], orig_img.shape[0]))

        # 5. Overlay
        superimposed = cv2.addWeighted(orig_img, 0.6, heatmap, 0.4, 0)

        # 6. Plot
        ai_verdict = "DEFECT" if score > threshold else "Good"
        text_color = "red" if ai_verdict == "DEFECT" else "green"
        filename = os.path.basename(img_path)[:20]

        axes[i].imshow(superimposed)
        
        axes[i].set_title(f"{filename}\n{ai_verdict} ({score.item():.4f})",
                          fontsize=10, color=text_color, fontweight='bold')
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()

# Run it
# MODEL_PATH = "results/patchcore/mvtec/run/weights/torch/model.pt" 
# TEST_IMAGES_DIR = "/kaggle/working/pv_organized_dataset_crack/test/defect"
# TEST_IMAGES_DIR = "/kaggle/input/test-microcrack/all_micro_crack"
TEST_IMAGES_DIR = "/kaggle/working/pv_organized_dataset_crack/test/good"

verify_with_heatmap_overlay(MODEL_PATH, TEST_IMAGES_DIR)



import torch
import glob
import os
import numpy as np
from tqdm import tqdm
from anomalib.deploy import TorchInferencer
from anomalib.data.utils import read_image

# === CONFIGURATION ===
# MODEL_PATH = "results/patchcore/mvtec/run/weights/torch/model.pt" # Update this
DEFECT_DIR = "/kaggle/working/pv_organized_dataset_crack/test/good" # Update this

def find_global_max_score():
    print("🚀 Scanning dataset to find Max Anomaly Score...")
    
    inferencer = TorchInferencer(path=MODEL_PATH, device="cpu")
    image_paths = glob.glob(os.path.join(DEFECT_DIR, "*.jpg")) + \
                  glob.glob(os.path.join(DEFECT_DIR, "*.png"))
    
    # Store the max score found in each image
    max_scores = []
    
    for path in tqdm(image_paths):
        try:
            image = read_image(path)
            result = inferencer.predict(image)
            
            # Get the anomaly map (pixel-wise scores)
            # This is a 2D map of float values
            anom_map = result.anomaly_map.squeeze().cpu().numpy()
            
            # Record the highest value in this specific image
            max_scores.append(anom_map.max())
        except Exception as e:
            print(f"Skipping {path}: {e}")

    if not max_scores:
        print("No images processed.")
        return 0.0

    # === STATISTICS ===
    max_scores = np.array(max_scores)
    true_max = np.max(max_scores)
    p99 = np.percentile(max_scores, 97) # 99th percentile (ignores extreme outliers)
    mean_max = np.mean(max_scores)

    print("\n" + "="*40)
    print("📊 GLOBAL SCORE STATISTICS")
    print("="*40)
    print(f"Highest Peak Observed (True Max): {true_max:.4f}")
    print(f"99th Percentile (Recommended):    {p99:.4f}")
    print(f"Average Peak Score:               {mean_max:.4f}")
    print("="*40)
    
    print(f"\n✅ SUGGESTION: Set GLOBAL_MAX_SCORE = {p99:.4f}")
    print("   (Using the 99th percentile usually gives the best contrast)")

if __name__ == "__main__":
    find_global_max_score()



def production_predict(image_path, model_path, threshold=0.82):
    """Production-ready prediction with optimal threshold."""
    from anomalib.deploy import TorchInferencer
    from anomalib.data.utils import read_image
    
    inferencer = TorchInferencer(path=model_path, device="cpu")
    image = read_image(image_path)
    predictions = inferencer.predict(image)
    
    score = predictions['pred_score'].item()
    is_defect = score > threshold
    
    # Calculate safety margin
    margin = abs(score - threshold)
    
    # Confidence levels
    if margin > 0.2:
        confidence = "VERY_HIGH"
    elif margin > 0.1:
        confidence = "HIGH"
    elif margin > 0.05:
        confidence = "MEDIUM"
    else:
        confidence = "BORDERLINE"
    
    # Generate report
    print("\n" + "="*50)
    print(f"🔍 SOLAR PANEL INSPECTION")
    print("="*50)
    print(f"File: {Path(image_path).name}")
    print(f"Score: {score:.4f}")
    print(f"Threshold: {threshold:.4f}")
    print(f"Margin: {margin:.4f}")
    print(f"Defect: {'🚨 YES' if is_defect else '✅ NO'}")
    print(f"Confidence: {confidence}")
    
    if is_defect and confidence == "BORDERLINE":
        print("⚠️  Borderline defect - recommend manual review")
    elif not is_defect and score > 0.7:
        print("ℹ️  Elevated score - monitor panel condition")
    
    print("="*50)
    
    return {
        'score': score,
        'is_defect': is_defect,
        'confidence': confidence,
        'threshold': threshold
    }

# Test with optimal threshold
result = production_predict(
    "/kaggle/input/test-anomaly-cracks/crack/000014_jpg.rf.ce1dabe2f2037486d170a4cd1c0516be_cell_127.jpg",
    # "/kaggle/working/pv_organized_dataset_crack/test/good/DSC_0496_cell18.jpg",
                            MODEL_PATH,
                            threshold=threshold)

File: 000014_jpg.rf.ce1dabe2f2037486d170a4cd1c0516be_cell_127.jpg
Score: 1.0000
Threshold: 0.8100
Margin: 0.1900
Defect: 🚨 YES
Confidence: HIGH

import zipfile
import os
from IPython.display import FileLink

# Assuming your folder is 'my_results_folder' in the working directory
zip_file_name = 'patchcore_model_robust_final_auc_99_f1_98_export.zip'
folder_to_zip = '/kaggle/working/patchcore_model_robust/weights/torch' # Or '/kaggle/working/my_results_folder'
folder_to_zip = '/kaggle/working/patchcore_model_robust/weights/torch'
# Create the zip file
with zipfile.ZipFile(zip_file_name, 'w', zipfile.ZIP_DEFLATED) as zipf:
    for root, dirs, files in os.walk(folder_to_zip):
        for file in files:
            if ".pt" in file or '.ckpt' in file:
                zipf.write(os.path.join(root, file), os.path.relpath(os.path.join(root, file), folder_to_zip))

# Display download link (optional but helpful)
display(FileLink(zip_file_name))


import anomalib
anomalib.__version__

#!/usr/bin/env python3
# standalone_inference.py

import os
os.environ['TRUST_REMOTE_CODE'] = '1'

import torch
import glob
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
from anomalib.deploy import TorchInferencer
from anomalib.data.utils import read_image

def load_model(model_path):
    """Load the exported Patchcore model."""
    print(f"Loading model from: {model_path}")
    
    inferencer = TorchInferencer(
        path=model_path,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    print("✅ Model loaded successfully!")
    return inferencer

def predict_single_image(inferencer, image_path, visualize=True):
    """Predict anomaly score for a single image."""
    # Read image
    image = read_image(image_path)
    
    # Predict
    predictions = inferencer.predict(image)
    
    # Extract score and label
    score = extract_value(predictions['pred_score'])
    label = extract_value(predictions['pred_label'])
    
    print(f"\n📊 Results for: {Path(image_path).name}")
    print(f"   Anomaly Score: {score:.4f}")
    print(f"   Prediction: {'🚨 DEFECT' if label == 1 else '✅ NORMAL'}")
    print(f"   Label: {label}")
    
    # Visualize if requested
    if visualize and "heat_map" in predictions:
        visualize_results(image_path, predictions, score, label)
    
    return {
        'image_path': image_path,
        'score': score,
        'label': label,
        'prediction': 'DEFECT' if label == 1 else 'NORMAL',
        'heatmap': predictions.get('heat_map', None)
    }

def extract_value(tensor_or_value):
    """Extract Python value from tensor or scalar."""
    if isinstance(tensor_or_value, torch.Tensor):
        return tensor_or_value.item() if tensor_or_value.numel() == 1 else float(tensor_or_value)
    return float(tensor_or_value) if isinstance(tensor_or_value, (int, float)) else tensor_or_value

def visualize_results(image_path, predictions, score, label):
    """Visualize the inference results with heatmap."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image
    axes[0].imshow(predictions["image"])
    axes[0].set_title("Original Image")
    axes[0].axis('off')
    
    # Heatmap
    heatmap = predictions["heat_map"]
    if isinstance(heatmap, torch.Tensor):
        heatmap = heatmap.cpu().numpy()
    
    heatmap_img = axes[1].imshow(heatmap, cmap='hot')
    axes[1].set_title("Anomaly Heatmap")
    axes[1].axis('off')
    plt.colorbar(heatmap_img, ax=axes[1], fraction=0.046, pad=0.04)
    
    # Overlay
    axes[2].imshow(predictions["image"])
    axes[2].imshow(heatmap, cmap='hot', alpha=0.5)
    axes[2].set_title(f"Overlay\nScore: {score:.3f}")
    axes[2].axis('off')
    
    plt.suptitle(f"Result: {'🚨 DEFECT' if label == 1 else '✅ NORMAL'} - {Path(image_path).name}", 
                fontsize=16)
    plt.tight_layout()
    plt.show()

def batch_inference(model_path, image_dir, output_csv=None):
    """Run inference on all images in a directory."""
    from tqdm import tqdm
    import pandas as pd
    
    # Load model
    inferencer = load_model(model_path)
    
    # Get all images
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(glob.glob(f"{image_dir}/**/{ext}", recursive=True))
    
    print(f"\nFound {len(image_paths)} images to process...")
    
    results = []
    for img_path in tqdm(image_paths, desc="Processing images"):
        try:
            image = read_image(img_path)
            predictions = inferencer.predict(image)
            
            score = extract_value(predictions['pred_score'])
            label = extract_value(predictions['pred_label'])
            
            results.append({
                'filename': Path(img_path).name,
                'path': img_path,
                'score': score,
                'label': label,
                'prediction': 'DEFECT' if label == 1 else 'NORMAL'
            })
            
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            results.append({
                'filename': Path(img_path).name,
                'path': img_path,
                'score': None,
                'label': None,
                'prediction': 'ERROR',
                'error': str(e)
            })
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Print summary
    print("\n" + "="*60)
    print("📊 INFERENCE SUMMARY")
    print("="*60)
    print(f"Total images processed: {len(df)}")
    
    if 'label' in df.columns and df['label'].notna().any():
        defect_count = (df['label'] == 1).sum()
        normal_count = (df['label'] == 0).sum()
        print(f"Normal predictions: {normal_count}")
        print(f"Defect predictions: {defect_count}")
        print(f"Defect rate: {defect_count/len(df)*100:.1f}%")
    
    # Save to CSV if requested
    if output_csv:
        df.to_csv(output_csv, index=False)
        print(f"\n✅ Results saved to: {output_csv}")
    
    return df

def main():
    parser = argparse.ArgumentParser(description="Patchcore Anomaly Detection Inference")
    parser.add_argument('--model', type=str, required=True,
                       help='Path to exported model (.pt file)')
    parser.add_argument('--image', type=str,
                       help='Path to single image')
    parser.add_argument('--dir', type=str,
                       help='Directory containing images for batch inference')
    parser.add_argument('--output', type=str, default='results.csv',
                       help='Output CSV file for batch results (default: results.csv)')
    parser.add_argument('--no-visualize', action='store_true',
                       help='Disable visualization')
    
    args = parser.parse_args()
    
    # Check if model exists
    if not os.path.exists(args.model):
        print(f"❌ Model not found: {args.model}")
        return
    
    if args.image:
        # Single image inference
        inferencer = load_model(args.model)
        result = predict_single_image(
            inferencer, 
            args.image, 
            visualize=not args.no_visualize
        )
        
    elif args.dir:
        # Batch inference
        if not os.path.exists(args.dir):
            print(f"❌ Directory not found: {args.dir}")
            return
        
        df = batch_inference(args.model, args.dir, args.output)
        
        # Show top 5 most anomalous
        if not df.empty and 'score' in df.columns:
            print("\n" + "="*60)
            print("🚨 TOP 5 MOST ANOMALOUS IMAGES")
            print("="*60)
            top_defects = df.nlargest(5, 'score')[['filename', 'score', 'prediction']]
            for idx, row in top_defects.iterrows():
                print(f"{row['filename']}: Score={row['score']:.4f} ({row['prediction']})")
    
    else:
        print("❌ Please specify either --image or --dir")
        parser.print_help()

if __name__ == "__main__":
    main()

# def train_simple_patchcore():
#     """Simple training without complex validation."""
    
#     # Create dataset
#     create_max_coverage_dataset()
    
#     # Use the SUBSET_DATA_ROOT directly with train/test only
#     DATASET_ROOT = "/kaggle/working/simple_dataset"
    
#     if os.path.exists(DATASET_ROOT):
#         shutil.rmtree(DATASET_ROOT)
    
#     # Copy only train and test (no val folder)
#     shutil.copytree(f"{SUBSET_DATA_ROOT}/train", f"{DATASET_ROOT}/train")
#     shutil.copytree(f"{SUBSET_DATA_ROOT}/test", f"{DATASET_ROOT}/test")
    
#     # Simple transforms
#     transform = v2.Compose([
#         v2.Resize((IMG_SIZE, IMG_SIZE)),
#         v2.ToImage(),
#         v2.ToDtype(torch.float32, scale=True),
#         v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#     ])
    
#     # Simple datamodule
#     datamodule = Folder(
#         root=DATASET_ROOT,
#         # image_size=(224, 224),
#         train_batch_size=16,
#         eval_batch_size=16,
#         num_workers=2,
#         task="classification",
#         # No validation split
#         val_split_ratio=0.0,
#     )
#     datamodule.train_transform = transform
#     datamodule.eval_transform = transform
    
#     # Simple model
#     model = Patchcore(
#         backbone="resnet18",
#         pre_trained=True,
#         layers=["layer2"],
#         coreset_sampling_ratio=0.05,
#     )
    
#     # Simple engine
#     engine = Engine(
#         max_epochs=1,
#         accelerator="auto",
#         enable_progress_bar=True,
#     )
    
#     print("Training simple model...")
#     engine.fit(model=model, datamodule=datamodule)
    
#     # Export
#     exported_path = engine.export(
#         model=model,
#         export_type="torch",
#         export_root=EXPORT_DIR
#     )
    
#     return engine, model, datamodule


# # Complete reset and install
# !pip install --upgrade pip setuptools wheel

# # Install specific compatible versions
# !pip install numpy==1.24.0
# !pip install torch==2.3.0 torchvision==0.18.0 --index-url https://download.pytorch.org/whl/cpu
# !pip install anomalib==2.2.0

# # Verify
# import numpy as np
# import torch
# print(f"NumPy: {np.__version__}")
# print(f"PyTorch: {torch.__version__}")
# print(f"Torchvision: {torchvision.__version__}")

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
import cv2
from pathlib import Path
import albumentations as A
from albumentations.pytorch import ToTensorV2

# ============================================
# 1. DATA HANDLING & AUGMENTATION
# ============================================

class PVCellDataset(Dataset):
    """Dataset for PV cells with heavy augmentation"""
    def __init__(self, image_paths, labels=None, transform=None, is_training=True):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.is_training = is_training
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        
        if image is None:
            raise ValueError(f"Failed to load image: {img_path}")
        
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
        
        # Return image and label if available
        if self.labels is not None:
            return image, self.labels[idx]
        return image

def get_augmentation_pipeline(is_training=True):
    """Heavy augmentation for limited healthy data"""
    if is_training:
        return A.Compose([
            A.Resize(IMG_SIZE, IMG_SIZE),
            # Geometric transforms
            A.Rotate(limit=15, p=0.7),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=10, p=0.5),
            
            # Intensity transforms (crucial for EL images)
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.7),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
            A.GaussianBlur(blur_limit=(3, 5), p=0.2),
            
            # Normalize
            A.Normalize(mean=[0.5], std=[0.5]),
            ToTensorV2(),
        ])
    else:
        return A.Compose([
            A.Resize(IMG_SIZE, IMG_SIZE),
            A.Normalize(mean=[0.5], std=[0.5]),
            ToTensorV2(),
        ])

# ============================================
# 2. AUTOENCODER MODEL
# ============================================

class ConvAutoencoder(nn.Module):
    """Convolutional Autoencoder for anomaly detection"""
    def __init__(self, latent_dim=128):
        super(ConvAutoencoder, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 4, stride=2, padding=1),  # 128x128
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(32, 64, 4, stride=2, padding=1),  # 64x64
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(64, 128, 4, stride=2, padding=1),  # 32x32
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(128, 256, 4, stride=2, padding=1),  # 16x16
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(256, latent_dim, 4, stride=2, padding=1),  # 8x8
            nn.BatchNorm2d(latent_dim),
            nn.LeakyReLU(0.2),
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 256, 4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            
            nn.ConvTranspose2d(32, 1, 4, stride=2, padding=1),
            nn.Tanh()
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded, encoded

# ============================================
# 3. FEATURE EXTRACTION (PRETRAINED)
# ============================================

class FeatureExtractor:
    """Extract features using pretrained ResNet"""
    def __init__(self, device='cuda'):
        self.device = device
        # Use pretrained ResNet18
        resnet = models.resnet18(pretrained=True)
        # Modify first layer for grayscale
        resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # Remove final FC layer
        self.model = nn.Sequential(*list(resnet.children())[:-1])
        self.model.to(device)
        self.model.eval()
    
    def extract(self, images):
        with torch.no_grad():
            features = self.model(images)
            features = features.view(features.size(0), -1)
        return features.cpu().numpy()

# ============================================
# 4. TRAINING PIPELINE
# ============================================
  
    
        
class AnomalyDetector:
    def __init__(self, device='cuda'):
        self.device = device
        self.autoencoder = ConvAutoencoder().to(device)
        self.feature_extractor = FeatureExtractor(device)
        self.threshold = None
        self.healthy_feature_mean = None
        self.healthy_feature_cov = None
        
    def train_autoencoder(self, train_loader, epochs=100, lr=0.001):
        """Train autoencoder on healthy cells only"""
        optimizer = torch.optim.Adam(self.autoencoder.parameters(), lr=lr)
        criterion = nn.L1Loss()
        
        self.autoencoder.train()
        for epoch in range(epochs):
            total_loss = 0
            for images in train_loader:
                images = images.to(self.device)
                
                optimizer.zero_grad()
                reconstructed, _ = self.autoencoder(images)
                loss = criterion(reconstructed, images)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(train_loader):.4f}")
    
    def compute_reconstruction_error(self, images):
        """Compute reconstruction error for anomaly scoring"""
        self.autoencoder.eval()
        with torch.no_grad():
            reconstructed, _ = self.autoencoder(images)
            # Per-sample MSE
            error = torch.mean((images - reconstructed) ** 2, dim=[1, 2, 3])
        return error.cpu().numpy()
    
    def compute_mahalanobis_distance(self, images):
        """Compute Mahalanobis distance in feature space"""
        features = self.feature_extractor.extract(images)
        diff = features - self.healthy_feature_mean
        inv_cov = np.linalg.pinv(self.healthy_feature_cov)
        distances = np.sqrt(np.sum(diff @ inv_cov * diff, axis=1))
        return distances
    
    def fit_threshold(self, val_loader, percentile=95):
        """Compute threshold on validation healthy cells"""
        all_scores = []
        
        for images in val_loader:
            images = images.to(self.device)
            
            # Reconstruction error
            recon_error = self.compute_reconstruction_error(images)
            
            # Mahalanobis distance
            mahal_dist = self.compute_mahalanobis_distance(images)
            
            # Combined score
            combined_score = normalize_scores(recon_error, mahal_dist) # recon_error + 0.5 * mahal_dist
            
            all_scores.extend(combined_score)
        
        # Set threshold at percentile
        self.threshold = np.percentile(all_scores, percentile)
        print(f"Threshold set at {percentile}th percentile: {self.threshold:.4f}")
    
    def fit_feature_distribution(self, train_loader):
        """Fit Gaussian distribution on healthy features"""
        all_features = []
        
        for images in train_loader:
            images = images.to(self.device)
            features = self.feature_extractor.extract(images)
            all_features.append(features)
        
        all_features = np.vstack(all_features)
        self.healthy_feature_mean = np.mean(all_features, axis=0)
        self.healthy_feature_cov = np.cov(all_features.T) + np.eye(all_features.shape[1]) *  0.01 #1e-6
    
    def optimize_threshold_f1(self, val_loader):
        """Find threshold that maximizes F1-score on validation set with defects"""
        all_scores = []
        all_labels = []
        
        # First, let's see what val_loader returns
        print(f"Val loader sample type check...")
        
        # Get one batch to check structure
        sample_batch = next(iter(val_loader))
        print(f"Batch type: {type(sample_batch)}")
        if isinstance(sample_batch, (list, tuple)):
            print(f"Batch length: {len(sample_batch)}")
            print(f"First element type: {type(sample_batch[0])}")
        
        # Now process all batches
        for batch in val_loader:
            if isinstance(batch, (list, tuple)):
                # Batch contains (images, labels)
                images, labels = batch
            else:
                # Batch contains only images
                images = batch
                # Create dummy labels (0 for healthy since validation has mixed data)
                labels = torch.zeros(len(images))
            
            images = images.to(self.device)
            recon_error = self.compute_reconstruction_error(images)
            mahal_dist = self.compute_mahalanobis_distance(images)
            combined = recon_error + 0.5 * mahal_dist
            all_scores.extend(combined)
            all_labels.extend(labels.cpu().numpy())
        
        # Try different thresholds
        all_scores_array = np.array(all_scores)
        all_labels_array = np.array(all_labels)
        
        # Use more thresholds for better optimization
        thresholds = np.linspace(np.percentile(all_scores_array, 5), 
                                np.percentile(all_scores_array, 95), 200)
        
        best_f1 = 0
        best_threshold = 0
        best_precision = 0
        best_recall = 0
        
        for thresh in thresholds:
            preds = (all_scores_array > thresh).astype(int)
            
            # Handle case where there are no positives
            if np.sum(preds) == 0:
                continue
                
            precision, recall, f1, _ = precision_recall_fscore_support(
                all_labels_array, preds, average='binary', zero_division=0)
            
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = thresh
                best_precision = precision
                best_recall = recall
        
        if best_threshold == 0:  # Fallback if no threshold found
            best_threshold = np.percentile(all_scores_array, 90)
        
        self.threshold = best_threshold
        print(f"\nOptimal threshold: {best_threshold:.4f}")
        print(f"Precision: {best_precision:.4f}")
        print(f"Recall: {best_recall:.4f}")
        print(f"F1-Score: {best_f1:.4f}")
        
        return best_threshold, best_f1
    
    def predict(self, images):
        """Predict if cells are anomalous"""
        images = images.to(self.device)
        
        # Reconstruction error
        recon_error = self.compute_reconstruction_error(images)
        
        # Mahalanobis distance
        mahal_dist = self.compute_mahalanobis_distance(images)
        
        # Combined anomaly score
        anomaly_score = normalize_scores(recon_error, mahal_dist) # recon_error + 0.5 * mahal_dist
        
        # Binary prediction
        predictions = (anomaly_score > self.threshold).astype(int)
        
        return predictions, anomaly_score

    def normalize_scores(self, recon_errors, mahal_dists):
        recon_norm = (recon_errors - np.mean(recon_errors)) / np.std(recon_errors)
        mahal_norm = (mahal_dists - np.mean(mahal_dists)) / np.std(mahal_dists)
        return recon_norm + mahal_norm
        
# ============================================
# 5. MAIN TRAINING SCRIPT
# ============================================



# def main():
# Configuration
HEALTHY_DIR = Path("/kaggle/working/pv_organized_dataset_crack/train/good")
DEFECTIVE_DIR = Path("/kaggle/working/pv_organized_dataset_crack/test/defect")
BATCH_SIZE = 32
EPOCHS = 100
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load paths
healthy_paths = list(HEALTHY_DIR.glob("*.png")) + list(HEALTHY_DIR.glob("*.jpg"))
defective_paths = list(DEFECTIVE_DIR.glob("*.png")) + list(DEFECTIVE_DIR.glob("*.jpg"))

print(f"Total healthy cells: {len(healthy_paths)}")
print(f"Total defective cells: {len(defective_paths)}")

# Split healthy cells: 70% train, 30% for validation+test
train_paths, temp_paths = train_test_split(healthy_paths, test_size=0.3, random_state=42)

# Split defective cells: 50% for validation (threshold optimization), 50% for test
defect_val_paths, defect_test_paths = train_test_split(
    defective_paths, test_size=0.5, random_state=42
)

# Split healthy temp: 50% validation, 50% test
healthy_val_paths, test_healthy_paths = train_test_split(
    temp_paths, test_size=0.5, random_state=42
)

# Create VALIDATION set with BOTH healthy and defective cells
val_paths_combined = healthy_val_paths + defect_val_paths
val_labels_combined = [0] * len(healthy_val_paths) + [1] * len(defect_val_paths)

# Create TEST set
test_defective_paths = defect_test_paths  # The other half of defective cells

print(f"\nDataset splits:")
print(f"Train (healthy only): {len(train_paths)}")
print(f"Validation (healthy): {len(healthy_val_paths)}")
print(f"Validation (defective): {len(defect_val_paths)}")
print(f"Test (healthy): {len(test_healthy_paths)}")
print(f"Test (defective): {len(test_defective_paths)}")

# Create datasets
train_dataset = PVCellDataset(train_paths, transform=get_augmentation_pipeline(True), is_training=True)
val_dataset = PVCellDataset(val_paths_combined, labels=val_labels_combined, 
                            transform=get_augmentation_pipeline(False), is_training=False)
test_healthy_dataset = PVCellDataset(test_healthy_paths, transform=get_augmentation_pipeline(False), is_training=False)
test_defective_dataset = PVCellDataset(test_defective_paths, transform=get_augmentation_pipeline(False), is_training=False)

# Create dataloaders
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
test_healthy_loader = DataLoader(test_healthy_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
test_defective_loader = DataLoader(test_defective_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

# Initialize detector
detector = AnomalyDetector(device=DEVICE)

# Step 1: Train autoencoder on healthy cells
print("\n=== Training Autoencoder ===")
detector.train_autoencoder(train_loader, epochs=EPOCHS)

# Step 2: Fit feature distribution
print("\n=== Fitting Feature Distribution ===")
detector.fit_feature_distribution(train_loader)
    
    

# Step 3: OPTIMIZE threshold using validation set with defects
print("\n=== Optimizing Threshold (maximizing F1) ===")
detector.optimize_threshold_f1(val_loader, val_labels_combined)

# Step 4: Evaluate on test set
print("\n=== Evaluating on Test Set ===")

# Test on healthy cells
healthy_preds = []
healthy_scores = []
for images in test_healthy_loader:
    preds, scores = detector.predict(images)
    healthy_preds.extend(preds)
    healthy_scores.extend(scores)

# Test on defective cells
defective_preds = []
defective_scores = []
for images in test_defective_loader:
    preds, scores = detector.predict(images)
    defective_preds.extend(preds)
    defective_scores.extend(scores)

# Calculate metrics
y_true = [0] * len(healthy_preds) + [1] * len(defective_preds)
y_pred = healthy_preds + defective_preds
y_scores = healthy_scores + defective_scores

precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
auc = roc_auc_score(y_true, y_scores)

print(f"\n=== Results ===")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")
print(f"AUC-ROC: {auc:.4f}")
print(f"Healthy cells classified as healthy: {sum([1-p for p in healthy_preds])}/{len(healthy_preds)}")
print(f"Defective cells detected: {sum(defective_preds)}/{len(defective_preds)}")

# Save model
torch.save({
    'autoencoder': detector.autoencoder.state_dict(),
    'threshold': detector.threshold,
    'feature_mean': detector.healthy_feature_mean,
    'feature_cov': detector.healthy_feature_cov,
}, 'pv_anomaly_detector.pth')
print("\nModel saved as 'pv_anomaly_detector.pth'")

# main()

test_good = os.listdir('/kaggle/working/pv_organized_dataset_crack/test/good')

from skimage.exposure import match_histograms # You may need to pip install scikit-image

def get_inference_transforms():
    """
    Robust transforms: Pad to square instead of squashing, then normalize.
    """
    return A.Compose([
        # 1. Resize longest side to 320 (matches your training size)
        A.LongestMaxSize(max_size=320),
        
        # 2. Pad with black pixels to make it a perfect 320x320 square
        A.PadIfNeeded(
            min_height=320, 
            min_width=320, 
            border_mode=cv2.BORDER_CONSTANT, 
            value=0
        ),
        
        # 3. Standard Normalization
        A.Normalize(mean=[0.5], std=[0.5]),
        ToTensorV2(),
    ])

def preprocess_image(image_path, reference_path=None):
    # 1. Load the Test Image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Could not load image at {image_path}")

    # 2. OPTIONAL BUT RECOMMENDED: Histogram Matching
    # This aligns the brightness/contrast of the test image to a 'gold standard' training image
    if reference_path:
        ref_img = cv2.imread(reference_path, cv2.IMREAD_GRAYSCALE)
        if ref_img is not None:
            # Match histograms
            image = match_histograms(image, ref_img, channel_axis=None)
            image = image.astype('uint8') # Convert back to uint8 for Albumentations

    # 3. Apply Geometric Transforms (Padding + Resizing)
    transform = get_inference_transforms()
    augmented = transform(image=image)
    image_tensor = augmented['image']
    
    # 4. Add Batch Dimension
    image_tensor = image_tensor.unsqueeze(0)
    
    return image_tensor, image


if __name__ == "__main__":
    # Settings
    CHECKPOINT_PATH = 'pv_anomaly_detector.pth'
    
    # 1. The image you want to test
    TEST_IMAGE_PATH = '/kaggle/working/pv_organized_dataset_crack/test/good/' + test_good[-2]
    
    # 2. A "Golden" image from your training set (Pick one good file and stick with it)
    REFERENCE_IMAGE_PATH = '/kaggle/working/pv_organized_dataset_crack/train/good/your_best_training_image.jpg' 
    
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    try:
        model = load_trained_model(CHECKPOINT_PATH, device=DEVICE)
        
        # Pass the reference path here!
        predict_single_image(model, TEST_IMAGE_PATH, device=DEVICE) # You'll need to update predict_single_image signature too
        
        # To make it easier, you can hardcode the reference loading inside preprocess_image 
        # or pass it manually like this:
        img_tensor, orig = preprocess_image(TEST_IMAGE_PATH, REFERENCE_IMAGE_PATH)
        img_tensor = img_tensor.to(DEVICE)
        
        # ... rest of prediction logic ...

    except Exception as e:
        print(f"Error: {e}")

import torch
import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2

# ============================================
# 1. SETUP & LOADING
# ============================================

def load_trained_model(checkpoint_path, device='cuda'):
    """
    Loads the trained model weights and statistical parameters 
    into the AnomalyDetector class.
    """
    print(f"Loading model from {checkpoint_path}...")
    
    # Initialize the detector structure
    detector = AnomalyDetector(device=device)
    
    # Load the saved dictionary
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # 1. Load Autoencoder weights
    detector.autoencoder.load_state_dict(checkpoint['autoencoder'])
    detector.autoencoder.eval() # Set to evaluation mode
    
    # 2. Load Statistical Params (Threshold, Mean, Covariance)
    detector.threshold = checkpoint['threshold']
    detector.healthy_feature_mean = checkpoint['feature_mean']
    detector.healthy_feature_cov = checkpoint['feature_cov']
    
    print("Model loaded successfully.")
    return detector

def preprocess_image(image_path):
    """
    Reads and preprocesses a single image to match the training format.
    """
    # Read as Grayscale (Crucial: Model expects 1 channel)
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    if image is None:
        raise ValueError(f"Could not load image at {image_path}")
        
    # Get the validation augmentation pipeline (No flips/rotations, just resize/norm)
    transform = get_augmentation_pipeline(is_training=False)
    
    # Apply transforms
    augmented = transform(image=image)
    image_tensor = augmented['image']
    
    # Add batch dimension (C, H, W) -> (1, C, H, W)
    image_tensor = image_tensor.unsqueeze(0)
    
    return image_tensor, image  # Return original for visualization if needed

# ============================================
# 2. RUNNING INFERENCE
# ============================================

def predict_single_image(model, image_path, device='cuda'):
    # 1. Preprocess
    img_tensor, original_img = preprocess_image(image_path)
    img_tensor = img_tensor.to(device)
    
    # 2. Predict
    # The predict method in your class returns (binary_pred, anomaly_score)
    # However, your class's predict method expects a batch. 
    # Since we passed a batch of size 1, we extract the first result.
    prediction, score = model.predict(img_tensor)
    
    # Extract scalar values
    is_defect = prediction[0]  # 1 for defect, 0 for healthy
    anomaly_score = score[0]
    
    # 3. Interpret results
    result_text = "DEFECTIVE" if is_defect == 1 else "HEALTHY"
    color = (0, 0, 255) if is_defect == 1 else (0, 255, 0) # Red or Green
    
    print(f"--- Inference Results ---")
    print(f"File: {image_path}")
    print(f"Prediction: {result_text}")
    print(f"Anomaly Score: {anomaly_score:.4f}")
    print(f"Threshold: {model.threshold:.4f}")
    
    return result_text, anomaly_score

# ============================================
# 3. USAGE EXAMPLE
# ============================================

if __name__ == "__main__":
    # Settings
    CHECKPOINT_PATH = 'pv_anomaly_detector.pth'
    # TEST_IMAGE_PATH = '/kaggle/input/test-microcrack/all_micro_crack/1000_jpg.rf.035b9d6152c6b97a98973700db44c908_cell_103.jpg' # <--- Change this
    # TEST_IMAGE_PATH = '/kaggle/input/test-anomaly-cracks/crack/000012_png.rf.def346ddbda4bfc4276eb56393ef7023_cell_40.jpg'
    # TEST_IMAGE_PATH = '/kaggle/working/pv_organized_dataset_crack/test/good/1169_1173_1164_D211AABB3A20220052_png.rf.85e2a9c192353384d5b7fcce3cc6eafd_cell25.jpg'
    # TEST_IMAGE_PATH = '/kaggle/working/pv_organized_dataset_crack/test/good/340_jpg.rf.f304b9c3e63867e26871d34cceb07c85_cell118.jpg'
    # TEST_IMAGE_PATH = '/kaggle/working/pv_organized_dataset_crack/test/good/1414_png.rf.391ea86ada1e90bcd0a80fb6f91ae9a1_cell30.jpg'
    TEST_IMAGE_PATH = '/kaggle/working/pv_organized_dataset_crack/test/good/' + test_good[-2]
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    try:
        # Load model
        model = load_trained_model(CHECKPOINT_PATH, device=DEVICE)
        
        # Run inference
        predict_single_image(model, TEST_IMAGE_PATH, device=DEVICE)

        visualize_prediction(model, TEST_IMAGE_PATH, device=DEVICE)

    except FileNotFoundError:
        print("Error: Checkpoint file or Image file not found.")
    except Exception as e:
        print(f"An error occurred: {e}")





import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch import ToTensorV2

# ==========================================
# 1. SETUP HELPERS (Must match your training)
# ==========================================

def get_valid_transforms():
    """Exact same transforms used during validation"""
    return A.Compose([
        A.Resize(IMG_SIZE, IMG_SIZE),
        A.Normalize(mean=[0.5], std=[0.5]),
        ToTensorV2(),
    ])

def preprocess_image(image_path):
    # Load Image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Image not found at {image_path}")
    
    # Transform
    transform = get_valid_transforms()
    augmented = transform(image=image)
    img_tensor = augmented['image']
    
    # Add batch dimension (1, C, H, W)
    return img_tensor.unsqueeze(0), image

# ==========================================
# 2. THE VISUALIZATION FUNCTION
# ==========================================

def visualize_prediction(detector, image_path, device='cuda'):
    """
    Shows: Input Image | AI Reconstruction | The Difference (Error)
    """
    # Prepare data
    img_tensor, original_img = preprocess_image(image_path)
    img_tensor = img_tensor.to(device)
    
    # Run Inference
    detector.autoencoder.eval()
    with torch.no_grad():
        reconstructed, _ = detector.autoencoder(img_tensor)
        preds, score = detector.predict(img_tensor)

    # Convert tensors back to images for display
    # We multiply by 0.5 and add 0.5 to undo the Normalization (-1 to 1 -> 0 to 1)
    input_display = img_tensor.cpu().numpy()[0, 0] * 0.5 + 0.5
    recon_display = reconstructed.cpu().numpy()[0, 0] * 0.5 + 0.5
    
    # Calculate the Difference Map (The "Heatmap" of defects)
    diff = np.abs(input_display - recon_display)

    # --- PLOTTING ---
    plt.figure(figsize=(15, 5))
    
    # 1. Input
    plt.subplot(1, 3, 1)
    plt.title("1. Input Image")
    plt.imshow(input_display, cmap='gray')
    plt.axis('off')
    
    # 2. Reconstruction
    plt.subplot(1, 3, 2)
    plt.title("2. Model Reconstruction\n(What it thinks is 'Normal')")
    plt.imshow(recon_display, cmap='gray')
    plt.axis('off')
    
    # 3. Difference
    plt.subplot(1, 3, 3)
    plt.title(f"3. Anomaly Map\nScore: {score[0]:.4f} (Thresh: {detector.threshold:.2f})")
    plt.imshow(diff, cmap='jet', vmin=0, vmax=1.0) # 'jet' makes high error RED
    plt.colorbar()
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

# ==========================================
# 3. RUN IT
# ==========================================

# if __name__ == "__main__":
    # 1. Load your model using the function we created earlier
    # Make sure 'load_trained_model' and 'AnomalyDetector' class are imported or defined above!
    # from your_model_file import load_trained_model, AnomalyDetector # <--- UPDATE THIS IMPORT

    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # model = load_trained_model('pv_anomaly_detector.pth', device=device)
    
    # 2. Visualize a specific image
    # Replace with a path to a real defective image to test


reference_img

# Add this inside preprocess_image, before transforms
# if use_histogram_matching:
    # reference_img is a sample 'good' image from your training set

import albumentations as A

image =  A.histogram_matching(image, reference_img, blend_ratio=1.0)

import numpy

numpy.__version__

!python -c "import anomalib; print(anomalib.__version__)"

