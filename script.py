from ultralytics import YOLO
import cv2
import numpy as np
import os
from roboflow import Roboflow
import yaml
from collections import defaultdict 
import os
import glob
import shutil

def find_categories(yaml_path):
  if not yaml_path:
      print("❌ Error: No 'data.yaml' found. Did you run the Roboflow download code first?")
  else:
      # ==========================================
      # 2. READ CLASS NAMES FROM YAML
      # ==========================================
    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)

    # Handle different YAML formats (list vs dict)
    class_names = config.get('names')
    if isinstance(class_names, list):
        id_to_name = {i: name for i, name in enumerate(class_names)}
    else:
        id_to_name = class_names

    print(f"📋 Classes detected: {list(id_to_name.values())}")

    # ==========================================
    # 3. COUNT LABELS IN TEXT FILES
    # ==========================================
    # Initialize counters
    counts = defaultdict(lambda: {'train': 0, 'valid': 0, 'test': 0})

    # Get the root directory of the dataset
    dataset_root = os.path.dirname(yaml_path)

    # Check standard YOLO subfolders
    splits = ['train', 'valid', 'test']

    for split in splits:
        # Colab/Roboflow structure is usually 'train/labels'
        target_dir = os.path.join(dataset_root, split, 'labels')

        # Sometimes it is just 'train' if labels and images are mixed
        if not os.path.exists(target_dir):
            target_dir = os.path.join(dataset_root, split)

        if not os.path.exists(target_dir):
            if split == 'valid':
                target_dir = os.path.join(dataset_root, 'val', 'labels')
                if not os.path.exists(target_dir):
                    print(f"⚠️ Warning: Could not find labels for '{split}'")
                    continue
            else:
                continue

        # Scan files
        txt_files = glob.glob(os.path.join(target_dir, "*.txt"))
        print(f"   Scanning {len(txt_files)} files in '{split}'...")

        for txt_file in txt_files:
            with open(txt_file, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) > 0:
                        class_id = int(parts[0])
                        # Get name or fallback to ID if missing
                        name = id_to_name.get(class_id, f"Unknown_ID_{class_id}")

                        # Store count (normalize 'val' to 'valid' for the table)
                        display_split = 'valid' if split == 'val' else split
                        counts[name][display_split] += 1

    # ==========================================
    # 4. PRINT PRETTY TABLE
    # ==========================================
    print("\n" + "="*65)
    print(f"{'Class Name':<30} {'Train':<10} {'Valid':<10} {'Test':<10}")
    print("-" * 65)

    # Sort by Train count high -> low
    sorted_classes = sorted(counts.items(), key=lambda x: x[1]['train'], reverse=True)

    for class_name, split_counts in sorted_classes:
        train_c = split_counts.get('train', 0)
        valid_c = split_counts.get('valid', 0)
        test_c  = split_counts.get('test', 0)
        print(f"{class_name[:30] if len(class_name)<30 else class_name[:27]+'...'  :<30} {train_c:<10} {valid_c:<10} {test_c:<10}")

    print("="*65)
    


# rf = Roboflow(api_key="60q1gKdq4vPi4tRgd831")
# project = rf.workspace("digitizo").project("pv-last-version-2-e46pq-x4fox")
# version = project.version(2)
# dataset = version.download("yolov8")


# dataset_path = dataset.location
# yaml_file = f"{dataset_path}/data.yaml"


# # 2. Define your rules using NAMES
# # Format: 'Old Name': 'New Target Name' (or None to delete)

# new_class_names = [
#     'examined',         # ID 0
#      'micro_crack',      # ID 1
#     'crack',            # ID 2
#     'low_cell',         # ID 3
#     'isolated_area',    # ID 4
#     'contamination',    # ID 5
#     'other_error'       # ID 6
# ]
# # DROP_CLASS_NAME = "other_error"
# DROP_CLASS_NAME = ""

# name_map = {
#     # 'Contamination': None,          # Delete this class
#     # 'Scratch': None,          # Delete this class
#     # Partionally Dark not added
#     #'Poor Ribbon Soldering': None,
#     'DarkSpot': None,
    
#     # Merge into 'examined'
#     "0 Examined": "examined",
#     "Examined": "examined",
#     "examined": "examined",

#     # Merge into 'micro_crack'
#     "Micro crack": "micro_crack",
#     "MicroCrack": "micro_crack",
#     "4 MicroCrack": "micro_crack",
#     "Microcracks": "micro_crack",

#     # Merge into 'crack'
#     "3 Crack": "crack", 
#     # "Backsheet Scratch": "crack", # cannot be combined
#     "Branch Cracks": "crack", # it is different
#     "Branch crack": "crack", # to check

#     # Merge into 'low_cell'
#     "2 ShortCircuitCell -LowPowerCell-": "low_cell",
#     "Dark Cell": "low_cell",
#     "black_cell": "low_cell",

#     "black_edge": "low_cell", # to check . on edge only . MAY BE REMOVED
    
#     #"DarkSpot": "low_cell", # cannot be full cell

#     # Mergo into 'isolated_area'
#     "Isolated area": "isolated_area",
#     "Isolation Area": "isolated_area",

#     # Mergo into 'contamination'
#     "Contamination": "contamination",

#     # group othererror
#     "5 OtherError": "other_error",
#     "Fingerline Interruption": "other_error",
#     "Shunt": "other_error",
#     "Scratch": "other_error",
#     "Backsheet Scratch": "other_error",
#     "Poor Ribbon Soldering": "other_error",
#     'Partially Dark': "other_error",

#     # "break": "other_error", # keep it away
# }


# # ==========================================
# # 0. CONFIGURATION & SETUP
# # ==========================================

# # PATHS
# ORIGINAL_DATASET_PATH = 'pv-last-version--2-2'
# NEW_DATASET_PATH = 'modified_data_1'   

# # ==========================================
# # 1. SETUP & COPY
# # ==========================================
# if os.path.exists(NEW_DATASET_PATH):
#     shutil.rmtree(NEW_DATASET_PATH)
# print(f"Copying dataset to {NEW_DATASET_PATH}...")
# shutil.copytree(ORIGINAL_DATASET_PATH, NEW_DATASET_PATH)

# yaml_file = os.path.join(NEW_DATASET_PATH, 'data.yaml')
# with open(yaml_file, 'r') as f:
#     config = yaml.safe_load(f)

# # Load old names
# old_class_names = config['names']
# if isinstance(old_class_names, list):
#     old_classes_dict = {idx: name for idx, name in enumerate(old_class_names)}
# else:
#     old_classes_dict = {idx: name for idx, name in old_class_names.items()}

# # ==========================================
# # 2. GENERATE ID MAP
# # ==========================================
# final_class_names = list(new_class_names)
# id_map = {}

# print("\n--- Mapping Rules ---")

# for old_id, old_name in old_classes_dict.items():
    
#     # 1. Check if we have a specific rule in name_map
#     if old_name in name_map:
#         target_name = name_map[old_name]
        
#         # If target is None, we delete it (if you ever need that)
#         if target_name is None:
#             id_map[old_id] = -1
#             print(f"DEL: '{old_name}' (ID {old_id}) -> Marked for deletion")
#             continue

#         # Check if target is already in our list
#         if target_name in final_class_names:
#             target_id = final_class_names.index(target_name)
#             id_map[old_id] = target_id
#             print(f"✅ MAP: '{old_name}' (ID {old_id}) -> Merged to '{target_name}' (ID {target_id})")
#         else:
#             # If target class doesn't exist yet, create it
#             final_class_names.append(target_name)
#             target_id = len(final_class_names) - 1
#             id_map[old_id] = target_id
#             print(f"🆕 NEW TARGET: '{old_name}' (ID {old_id}) -> Added new class '{target_name}' (ID {target_id})")
            
#     # 2. CATCH-ALL: If NOT in name_map, KEEP IT!
#     else:
#         # Check if this name already exists in the final list (to avoid duplicates)
#         if old_name in final_class_names:
#             new_id = final_class_names.index(old_name)
#             id_map[old_id] = new_id
#             print(f"🔸 AUTO-MERGE: '{old_name}' (ID {old_id}) -> Exists at ID {new_id}")
#         else:
#             # It's a completely new class (like 'break'), add it to the list
#             final_class_names.append(old_name)
#             new_id = len(final_class_names) - 1
#             id_map[old_id] = new_id
#             print(f"🚀 AUTO-KEEP: '{old_name}' (ID {old_id}) -> Kept as NEW ID {new_id}")

# # ==========================================
# # 3. REWRITE LABELS
# # ==========================================
# print("\n--- Processing Labels ---")
# for subdir in ['train', 'valid', 'test']:
#     labels_path = os.path.join(NEW_DATASET_PATH, subdir, 'labels')
#     if not os.path.exists(labels_path): continue
    
#     for filename in os.listdir(labels_path):
#         if not filename.endswith('.txt'): continue
#         file_path = os.path.join(labels_path, filename)
        
#         with open(file_path, 'r') as f:
#             lines = f.readlines()
        
#         new_lines = []
#         file_changed = False
        
#         for line in lines:
#             parts = line.strip().split()
#             if not parts: continue
            
#             try:
#                 curr = int(parts[0])
#             except:
#                 continue

#             if curr in id_map:
#                 new_id = id_map[curr]
#                 if new_id != -1:
#                     if new_id != curr:
#                         parts[0] = str(new_id)
#                         new_lines.append(" ".join(parts) + "\n")
#                         file_changed = True
#                     else:
#                         new_lines.append(line)
#                 else:
#                     file_changed = True # Deleted line
#             else:
#                 # Should not happen with Auto-Keep logic, but safe deletion if somehow missed
#                 file_changed = True 
        
#         if file_changed:
#             with open(file_path, 'w') as f:
#                 f.writelines(new_lines)

# # ==========================================
# # 4. SAVE YAML
# # ==========================================
# config['names'] = final_class_names
# config['nc'] = len(final_class_names)
# config['path'] = NEW_DATASET_PATH

# with open(yaml_file, 'w') as f:
#     yaml.dump(config, f)

# print("\n========================================")
# print(f"Final Class List ({len(final_class_names)} classes):")
# print(final_class_names)
# print("========================================")


# find_categories(yaml_file)


if __name__ == '__main__':
    yaml_file = r'dataset_v2_960_tiled\data.yaml'

    # from multiprocessing import freeze_support

    # freeze_support()

    model = YOLO('yolov8m.pt')

    results = model.train(
        data=yaml_file,
        epochs=50,
        imgsz=640,
        batch=6,
        patience=20,
        device=0,
        plots=True
    )

    