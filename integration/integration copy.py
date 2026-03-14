from pdf_report import create_pdf_report
from anomaly_model import AnomalyDetector, load_trained_model, predict_single_image

import cv2
import numpy as np
from ultralytics import YOLO
import logging
from collections import Counter
import torch
from anomalib.deploy import TorchInferencer

import albumentations as A
from albumentations.pytorch import ToTensorV2

import os
os.environ['TRUST_REMOTE_CODE'] = "1" # for loading anomaly model

import sys
import pathlib

# Fix for loading models created on Linux/Posix systems on Windows
# This allows PosixPath to be instantiated as a WindowsPath
if sys.platform == 'win32':
    pathlib.PosixPath = pathlib.WindowsPath


# Setup Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("PV_Pipeline")

class PVInspectionPipeline:

    def __init__(self, pv_seg_model_path, cell_detector_model_path, detector_model_imgsz, anomaly_model_path, defect_model_path, defect_detector_model_imgsz):
        """
        Initialize models once to save time during requests.
        """
        logger.info("Loading AI Models...")
        
        # self.pv_seg_model = YOLO(pv_seg_model_path)

        self.cell_detector_model = YOLO(cell_detector_model_path)
        self.detector_model_imgsz = detector_model_imgsz

        # 2. Load Anomaly Detector
        # self.anomaly_model = TorchInferencer(
        #     path=anomaly_model_path, 
        #     device="cuda" if torch.cuda.is_available() else "cpu"
        # )

        self.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

        # Load model
        self.anomaly_model, self.CELL_ANOM_THRESH = load_trained_model(anomaly_model_path, device=self.DEVICE)

        self.anomaly_model_path = anomaly_model_path

        # 1. Load Defect Detector (YOLOv8s - Cell Level)
        self.cell_defect_model = YOLO(defect_model_path)
        self.defect_detector_model_imgsz = defect_detector_model_imgsz
        
        logger.info("Pipeline Ready.")


    def analyze_image(self, image_path):
        """
        MAIN ENTRY POINT: Connects all steps from the flowchart.
        """
        report = {
            "status": "PASS",
            "final_score": 100,  # Starts at 100 (Perfect)
            "snr_value": 0,
            "failed_cells": [],  # List of {id, defect_type, score, crop_img}
            "image_path": image_path,
            "annotated_image": None
        }

        # --- STEP 1: Load Raw Image ---
        raw_img = cv2.imread(image_path)
        if raw_img is None:
            raise ValueError("Image not found")

        # --- STEP 2: PV Module Segmentation and Extraction (Perspective Transform) ---
        # "Straightens" the panel so cells are perfectly square
        # module_img = self._extract_pv_module(raw_img)
        module_img = raw_img

        # --- STEP 3: Calculate SNR (Parallel Branch 1) ---
        # Checks for image quality (too dark/noisy?)
        report["snr_value"] = self._calculate_snr(module_img)
        if report["snr_value"] < 5.0:
            logger.warning(f"Low SNR detected: {report['snr_value']:.2f}")
            # Logic: If image is garbage, maybe abort or flag warning?

        # --- STEP 4: Cell Extraction (Branch 2) ---
        # Slices the panel into individual cells
        cells, raw_pv_img = self._extract_cells(module_img) 

        # --- STEP 5 & 6: Anomaly & Defect Detection (The Loop) ---
        anomaly_scores = []
        # defect_confs = []
        defect_confs = []
        total_anomalies = 0

        for cell in cells:
            # Sub-Step A: Anomaly Classification (Is it weird?)
            # Deviation from clean reference
            anom_score = self._predict_anomaly(cell['crop'])
            anomaly_scores.append(anom_score)
            # YOLO prediction on this specific cell
            defects = self._detect_defects(cell['crop']) ### [IMP] may be placed for every cell (before cond) to double check with anomaly score
            
            # LOGIC: If Anomaly is High OR Defect Found
            if anom_score > self.CELL_ANOM_THRESH or len(defects) > 0: # IMP update . # it is changed in method 
                # Sub-Step B: Defect Detection (What is it?)
                total_anomalies += 1

                # Add to Report
                cell_data = {
                    "cell_id": cell['id'],
                    "pos": cell['pos_label'], # e.g., "R1-C3"
                    "row": cell['row'],
                    "col": cell['col'],
                    # "anomaly_score": anom_score,
                    "is_anomaly": anom_score > self.CELL_ANOM_THRESH ,
                    "defects": defects # List of found issues
                }

                if len(defects) == 0 and anom_score > self.CELL_ANOM_THRESH:
                    print('dfd', self.CELL_ANOM_THRESH)
                    cell_data["defects"].append({"type": "Unknown_Anomaly", "conf": anom_score})
                
                # Aggregate Score
                max_conf = max([d['conf'] for d in cell_data['defects']]) if cell_data['defects'] else anom_score
                defect_confs.append(max_conf)

                report["failed_cells"].append(cell_data)
            # print("Anomaly Cell: " , anom_score > self.CELL_ANOM_THRESH)

        report["annotated_image"] = self._draw_colored_pv(module_img, cells, report["failed_cells"])
        # Save the new colored image for debugging
        cv2.imwrite(f'test_colored_pv_{os.path.splitext(os.path.basename(image_path))[0]}.png', report["annotated_image"])

        # print("cell data:", report)

        # --- STEP 7: Final PV Defective Score Calculation ---
        report["final_score"] = self._calculate_final_score(
            snr=report["snr_value"],
            total_anomalies=total_anomalies,
            defect_confidences=defect_confs,
            num_cells=len(cells)
        )

        # Determine Final Pass/Fail
        if report["final_score"] < 80: # Threshold
            report["status"] = "FAIL"

        return report

    # ================= HELPER FUNCTIONS =================
    # done
    def _extract_pv_module(self, img):
        """
        Detects corners and warps perspective.
        For now, this is a placeholder that assumes the image is mostly centered.
        In production, use `cv2.findContours` to find the panel edges.
        """
        
        prediction = self.pv_seg_model.predict(img)
        # ...
        ## return final wrapped image
        return prediction

    # to see
    def _calculate_snr(self, img):
        """
        Signal-to-Noise Ratio: Mean / StdDev
        High SNR = Clear Image. Low SNR = Noisy/Grainy.
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        mean, std = cv2.meanStdDev(gray)
        if std[0][0] == 0: return 0
        return float(mean[0][0] / std[0][0])

    
    def _filter_cell_boxes_by_width_mode(self, boxes, tolerance=0.2):
        """
        Filters out outlier boxes based on width.
        Args:
            boxes: List of [x1, y1, x2, y2]
            tolerance: percentage (0.2 = 20%) allowed deviation from the mode width.
        """
        if not boxes:
            return []

        # 1. Calculate widths
        widths = [b[2] - b[0] for b in boxes]
        
        if not widths:
            return []

        # 2. Find the Mode (Most common width range)
        # We bin widths into histograms to find the "peak" frequency
        bins = np.linspace(min(widths), max(widths), num=20)
        hist, bin_edges = np.histogram(widths, bins=bins)
        
        # Get the index of the most frequent bin
        mode_idx = np.argmax(hist)
        mode_width = (bin_edges[mode_idx] + bin_edges[mode_idx+1]) / 2

        # 3. Filter
        filtered_boxes = []
        min_w = mode_width * (1 - tolerance)
        max_w = mode_width * (1 + tolerance)

        for b in boxes:
            w = b[2] - b[0]
            if min_w <= w <= max_w:
                filtered_boxes.append(b)
        
        return filtered_boxes
    
    # maybe done
    def _annotate_pv_by_cells(self, image, detections):
        img = image.copy()
        h, w, _ = img.shape

        # Draw boxes
        for (x1, y1, x2, y2) in detections:
            cv2.rectangle(
                img,
                (int(x1), int(y1)),
                (int(x2), int(y2)),
                (255, 0, 0),
                2
            )

        # Draw cell count banner
        banner_h = 50
        cv2.rectangle(img, (0, 0), (w, banner_h), (0, 0, 0), -1)

        text = f"Cells detected: {len(detections)}"
        cv2.putText(
            img,
            text,
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 255),
            2
        )

        return img

    def _sort_boxes_top_down_left_right(self, boxes, y_tolerance_ratio=0.5):
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

    
    def _sort_boxes_grid(self, boxes):
        """
        Sorts bounding boxes top-to-bottom, left-to-right.
        Handles slight misalignments (wobbly rows) using clustering.
        """
        if not boxes: return []
        
        # Calculate average cell height to define a "Row Threshold"
        # If two boxes are within 50% of a cell height vertically, they are in the same row.
        avg_height = np.mean([b[3] - b[1] for b in boxes])
        y_threshold = avg_height * 0.5 
        
        # 1. Sort primarily by Y (Top Edge)
        boxes_by_y = sorted(boxes, key=lambda b: b[1])
        
        rows = []
        current_row = []
        
        # 2. Cluster into Rows
        for box in boxes_by_y:
            if not current_row:
                current_row.append(box)
                continue
            
            # Compare with the first box in the current row
            ref_y = current_row[0][1]
            
            if abs(box[1] - ref_y) < y_threshold:
                current_row.append(box)
            else:
                # Finish current row, start a new one
                rows.append(current_row)
                current_row = [box]
        
        if current_row:
            rows.append(current_row)
            
        # 3. Sort each Row by X (Left Edge) and Assign Indices
        final_sorted = []
        for r_idx, row_boxes in enumerate(rows):
            # Sort this row left-to-right
            row_boxes.sort(key=lambda b: b[0])
            
            for c_idx, box in enumerate(row_boxes):
                # r_idx + 1 makes it 1-based (Row 1, not Row 0)
                final_sorted.append((box, r_idx + 1, c_idx + 1))
                
        return final_sorted
    

    # need to be updated
    def _extract_cells(self, pv_img):
        """
        Detects, sorts, and crops cells from the PV module image.
        Returns:
            cells (list of dicts): Each dict contains 'crop', 'row', 'col', 'id'.
            annotated_pv (ndarray): Image with drawn detections.
        """
        
        pv_h, pv_w = pv_img.shape[:2]
        
        # 1. Inference
        results = self.cell_detector_model.predict(
            pv_img, 
            conf=0.4, 
            iou=0.6,  
            imgsz=self.detector_model_imgsz, 
            verbose=False
        )

        boxes = results[0].boxes.xyxy.cpu().numpy()
        l_boxes = [list(map(int, b)) for b in boxes] # Convert floats to ints

        # 2. Filter Noise
        filtered_boxes = self._filter_cell_boxes_by_width_mode(l_boxes)

        # 3. Sort & Grid Assignment (Crucial Step)
        # This helper returns tuples: (box, row_idx, col_idx)
        sorted_boxes_with_pos = self._sort_boxes_grid(filtered_boxes)

        cells = []

        # 4. Process Sorted Boxes
        for box, row, col in sorted_boxes_with_pos:
            x1, y1, x2, y2 = map(int, box)
                        
            # Crop
            cell_crop = pv_img[y1:y2, x1:x2]

            if cell_crop.size == 0:
                continue

            # Create Position Label (e.g., "R1-C3")
            pos_label = f"R{row}-C{col}"
            
            # Store as a DICTIONARY (Better than a set for later use)
            cells.append({
                'id': len(cells) + 1,   # Sequential ID (1, 2, 3...)
                'pos_label': pos_label, # String ID
                'row': row,             # Int ID
                'col': col,             # Int ID
                'crop': cell_crop,      # The Image
                'box': [x1, y1, x2, y2] # Original coordinates
            })

        # 5. Annotate (Optional: Use sorted boxes to match visual order)
        # sorted_boxes_only = [c['box'] for c in cells]
        # annotated_pv = self._annotate_pv_by_cells(pv_img, sorted_boxes_only)

        return cells, pv_img
      
    
    def _predict_anomaly(self, cell_img):
        """
        Runs the Anomalib/PatchCore model.
        cell_img: as RGB
        Returns float 0.0 (Clean) to 1.0 (Highly Anomalous).
        """
        
        
        # Run inference
        anom_text, anom_score = predict_single_image(self.anomaly_model, self.CELL_ANOM_THRESH, cell_img, device=self.DEVICE)

        # visualize_prediction(model, TEST_IMAGE_PATH, device=DEVICE)

        # anom_score = float(self.anomaly_model.predict(image=cell_img).pred_score)

        return anom_score

    def _draw_colored_pv(self, image, all_cells, failed_cells_data):
        """
        Draws colored boxes over the PV image based on analysis results.
        - Red: Defects found
        - Orange: Anomaly found (but no defect label)
        - Green: Normal
        """
        img = image.copy()
        overlay = image.copy()
        
        # Create a lookup dict for failed cells for speed (O(1) access)
        failed_map = {item['cell_id']: item for item in failed_cells_data}
        
        h, w = img.shape[:2]

        for cell in all_cells:
            cid = cell['id']
            x1, y1, x2, y2 = cell['box']
            
            color = None
            label = ""
            
            # Check if this cell is in the failed list
            if cid in failed_map:
                data = failed_map[cid]
                
                # RULE 1: If specific defects found -> RED
                if len(data['defects']) > 0:
                    color = (0, 0, 255) # Red in BGR
                    # Get the defect name for the label
                    d_names = [d['type'] for d in data['defects']]
                    label = f"{d_names[0]}"
                
                # RULE 2: Anomaly only (High score but no defects) -> ORANGE
                elif data['is_anomaly']:
                    color = (0, 165, 255) # Orange in BGR
                    label = "Anomaly"
            
            # RULE 3: Normal/Healthy -> GREEN
            else:
                color = (0, 255, 0) # Green in BGR
                label = "OK"
            
            # Draw Filled Rectangle (Semi-transparent)
            # We draw on 'overlay' first
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
            
            # Draw Text Label
            cv2.putText(overlay, label, (x1 + 5, y1 + 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        # Blend the overlay with the original image (0.3 transparency)
        cv2.addWeighted(overlay, 0.4, img, 0.6, 0, img)
        
        # Draw Solid Borders (Thicker) on top of the blend
        for cell in all_cells:
            cid = cell['id']
            x1, y1, x2, y2 = cell['box']
            color = (0, 255, 0) # Default Green
            
            if cid in failed_map:
                data = failed_map[cid]
                if len(data['defects']) > 0:
                    color = (0, 0, 255) # Red
                elif data['is_anomaly']:
                    color = (0, 165, 255) # Orange
            
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        return img
    
    def _detect_defects(self, cell_img):
        """
        Runs YOLO on the single cell.
        """
        results = self.cell_defect_model(cell_img, verbose=False, conf=0.25, iou=0.4, imgsz=self.defect_detector_model_imgsz)
        detections = []
        
        for r in results:
            for box in r.boxes:
                detections.append({
                    "type": self.cell_defect_model.names[int(box.cls)],
                    "conf": float(box.conf),
                    "box": box.xyxy.tolist()
                })
    
        return detections

    def _calculate_final_score(self, snr, total_anomalies, defect_confidences, num_cells):
        """
        Create a weighted formula based on your business rules.
        """
        base_score = 100
        
        # Penalties
        # 1. Major Defect Penalty (Exponential decay based on confidence)
        defect_penalty = sum([conf * 20 for conf in defect_confidences]) 
        
        # 2. Anomaly Penalty (General weirdness across panel)
        anomaly_penalty = (total_anomalies / num_cells) * 50 
        
        # 3. SNR Penalty (If image is garbage, reduce trust score)
        snr_penalty = 10 if snr < 3.0 else 0
        
        final = base_score - defect_penalty - anomaly_penalty - snr_penalty
        return max(0, min(100, final)) # Clamp between 0 and 100

# ================= USAGE IN FLASK/WEB APP =================
if __name__ == "__main__":

    PV_SEG_MODEL = r"models/best.pt" # SAM or YOLO
    CELL_DETECTOR_MODEL = r"models/cell_detector.pt"
    CELL_DETECTOR_IMGS = 1280

    CELL_ANOMALY_MODEL = r"models/cell_anomaly.pt"
    CELL_ANOMALY_MODEL = r'models/pv_anomaly_detector.pth' # checkpoint

    DEFECT_MODEL = r"models/defect_detector.pt" # cell-level defect detection
    DEFECT_IMGS = 640
    
    # 1. Initialize logic (Do this when server starts)
    pipeline = PVInspectionPipeline(
        pv_seg_model_path=PV_SEG_MODEL,
        cell_detector_model_path=CELL_DETECTOR_MODEL, 
        detector_model_imgsz = CELL_DETECTOR_IMGS,
        anomaly_model_path=CELL_ANOMALY_MODEL,
        defect_model_path=DEFECT_MODEL,
        defect_detector_model_imgsz = DEFECT_IMGS,

    )
    
    # 2. Process an upload (Do this per request)
    TEST_IMG_PATH = r"test images\known_clean_DSC_1315.JPG"
    result_report = pipeline.analyze_image(TEST_IMG_PATH)
    
    print("Generating PDF Report...")
    create_pdf_report(result_report, f"PV_Inspection_Result_{os.path.splitext(os.path.basename(TEST_IMG_PATH))[0]}.pdf")

    # 3. Send JSON back to frontend
    print("Final JSON for Web App:", result_report)

