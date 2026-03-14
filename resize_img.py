import cv2
import os
import glob

# Settings
INPUT_FOLDER = r'cell_separation_from_mixed_orig\final_yolo_datasets\vendor_2\valid\images'
OUTPUT_FOLDER = r'cell_separation_from_mixed_orig\final_yolo_datasets\vendor_2_resized\valid\images'
TARGET_SIZE = (640, 640) # A safe square size if using P2 Head. Use 1280 if standard YOLO.

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Process
files = glob.glob(os.path.join(INPUT_FOLDER, '*.jpg')) # or .png

for f in files:
    img = cv2.imread(f)
    h, w = img.shape[:2]
    
    # Logic: Only upscale if image is small
    # For ELPV, we force upscale everything to ensure consistent texture
    
    # INTER_LANCZOS4 is crucial here. 
    # It creates artifacts (ringing) on sharp edges, which actually HELPS
    # the CNN detect the crack boundary.
    resized_img = cv2.resize(img, TARGET_SIZE, interpolation=cv2.INTER_LANCZOS4)
    
    # Save
    name = os.path.basename(f)
    cv2.imwrite(os.path.join(OUTPUT_FOLDER, name), resized_img)

print(f"Processed {len(files)} images using Lanczos upscaling.")