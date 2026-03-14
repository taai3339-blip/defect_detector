import os
import shutil
import glob
from ultralytics import YOLO

# 1. Load the mystery model
model = YOLO(r'C:\Users\Rowan\Documents\Final_demo\Final_demo\demo\demo\demo\demo\src\main\backend\models/best2.pt') 
# model = YOLO(r'models\cell_detector.pt') 
# model = YOLO(r'runs\detect\train54\weights\best.pt')

# 2. Define paths
TEST_IMAGES = r'C:\Users\Rowan\Documents\Rowan\all clean'
# TEST_IMAGES = r'modified_data_1\train\images\000006_jpg.rf.f4f12acc41af6b6a1350184861a75162.jpg'
OUTPUT_DIR = r'cell_detect_level_clean_for_anomaly_testing'
OUTPUT_DIR = r'C:\Users\Rowan\Documents\Rowan\Yolo_test\cell_model_test_results'

# 3. Clean up previous results
# if os.path.exists(OUTPUT_DIR):
#     shutil.rmtree(OUTPUT_DIR)

# 4. Run Inference
# conf=0.25 is standard. If boxes are missing, lower it to 0.10.
results = model.predict(
    source=TEST_IMAGES,
    project=OUTPUT_DIR,
    name='visuals',
    save=True,      # Save images with boxes
    # save_txt=True, # We just want to look for now
    conf=0.25,
    max_det=500     # Allow it to find up to 200 cells (standard panels have 60-96)
)

# print(results)
boxes = []
boxes.extend([len(i.boxes) for i in results])

# count =  len(results[0].boxes) # len(boxes)

print(f'Boxes: {boxes}')

print(f"Check the images in: {os.path.join(OUTPUT_DIR, 'visuals')}")



# model = YOLO('path/to/mystery_cell_model.pt')
# images = glob.glob(f'{TEST_IMAGES}/*.jpg') # Adjust extension if needed

# print(f"{'Image Name':<30} | {'Cells Found':<10} | {'Status'}")
# print("-" * 60)

# for img_path in images[:20]: # Check first 20 images
#     results = model.predict(img_path, conf=0.25, verbose=False)[0]
#     count = len(results.boxes)
#     print(count)
#     # Simple logic: Adjust "72" to whatever your real panel size is (e.g. 60, 96, 120)
#     EXPECTED_CELLS = 72 
    
#     if count == EXPECTED_CELLS:
#         status = "✅ Perfect"
#     elif abs(count - EXPECTED_CELLS) < 5:
#         status = "⚠️ Close"
#     else:
#         status = "❌ FAIL"
        
#     print(f"{os.path.basename(img_path):<30} | {count:<10} | {status}")
