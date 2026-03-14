import cv2
import numpy as np
from ultralytics import YOLO
import logging
from collections import Counter


# Setup Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("PV_Pipeline")

class PVInspectionPipeline:

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
        sorted_boxes_only = [c['box'] for c in cells]
        annotated_pv = self._annotate_pv_by_cells(pv_img, sorted_boxes_only)


        # for _, det in enumerate(sorted_boxes):
            # x1, y1, x2, y2 = map(int, det)
            
            
            # Calculate padded coordinates, clamping to image boundaries
            # x1_padded = max(0, x1 - CELL_PADDING)
            # y1_padded = max(0, y1 - CELL_PADDING)
            # x2_padded = min(pv_w, x2 + CELL_PADDING)
            # y2_padded = min(pv_h, y2 + CELL_PADDING)
            
            # Crop with padding
            # cell_crop = pv_img[y1_padded:y2_padded, x1_padded:x2_padded]

            # if cell_crop.size == 0:
            #     continue
            
            # cells.append({pos_in_pv, cell_crop}) # need pos_in_pv so it is presented in the report matrix with cell id . may be sorted from top left first

        # annotated_pv = self._annotate_pv_by_cells(pv_img, boxes)

        return cells, annotated_pv
    

        # done
    # def _filter_cell_boxes_by_width_mode(boxes, tolerance=0.05):
    #     """
    #     boxes: list of [x_min, y_min, x_max, y_max]
    #     tolerance: fraction deviation from mode (e.g., 0.3 → ±30%)
    #     """

    #     widths = [(int(x2)-int(x1)) for x1, y1, x2, y2 in boxes]
    #     if not widths:
    #         return []
        
    #     width_counts = Counter(widths)
    #     mode_width = width_counts.most_common(1)[0][0]
        

    #     filtered_boxes = []

    #     for box in boxes:
    #         x_min, y_min, x_max, y_max = map(int, box)
    #         width = x_max - x_min
    #         if abs(width - mode_width) / mode_width <= tolerance:
    #             filtered_boxes.append(box)

    #     return filtered_boxes


    def __init__(self, pv_seg_model_path, cell_detect_model_path, anomaly_model_path, cell_defect_model_path):
        """
        Initialize models once to save time during requests.
        """
        logger.info("Loading AI Models...")
        
        # 1. Load Defect Detector (YOLOv8s - Cell Level)
        self.pv_seg_model = YOLO(pv_seg_model_path)

        self.cell_detect_model = YOLO(cell_detect_model_path)

        # 2. Load Anomaly Detector (Optional: PatchCore/Anomalib)
        self.anomaly_model = PatchCore.load(anomaly_model_path)

        self.cell_defect_model = YOLO(cell_defect_model_path)
            
            
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
            "image_path": image_path
        }

        # --- STEP 1: Load Raw Image ---
        raw_img = cv2.imread(image_path)
        if raw_img is None:
            raise ValueError("Image not found")

        # --- STEP 2: PV Module Segmentation and Extraction (Perspective Transform) ---
        # "Straightens" the panel so cells are perfectly square
        module_img = self._extract_pv_module(raw_img)

        # --- STEP 3: Calculate SNR (Parallel Branch 1) ---
        # Checks for image quality (too dark/noisy?)
        report["snr_value"] = self._calculate_snr(module_img)
        if report["snr_value"] < 5.0:
            logger.warning(f"Low SNR detected: {report['snr_value']:.2f}")
            # Logic: If image is garbage, maybe abort or flag warning?

        # --- STEP 4: Cell Extraction (Branch 2) ---
        # Slices the panel into individual cells
        cells = self._extract_cells(module_img) 

        # --- STEP 5 & 6: Anomaly & Defect Detection (The Loop) ---
        anomaly_scores = []
        defect_confs = []

        for cell_id, (pos-in_pv, cell_img) in enumerate(cells):
            
            # Sub-Step A: Anomaly Classification (Is it weird?)
            # Deviation from clean reference
            anom_score = self._predict_anomaly(cell_img)
            anomaly_scores.append(anom_score)


            # LOGIC: If Anomaly is High OR Defect Found
            if anom_score > 0.5 or len(defects) > 0:
                # Sub-Step B: Defect Detection (What is it?)
                # YOLO prediction on this specific cell
                defects = self._detect_defects(cell_img) # may be placed for every cell (before cond) to double check with anomaly score

                # # Handling "Failed Detection / New Defect" logic
                # if len(defects) == 0 and anom_score > 0.8:
                #     cell_data["defects"].append({"type": "Unknown_Anomaly", "conf": anom_score})

                cell_data = {
                    "cell_id": cell_id,
                    "anomaly_score": anom_score,
                    "defects": defects # List of [cls, conf, box]
                }
        
                # Handling "Failed Detection / New Defect" logic from chart
                if len(defects) == 0 and anom_score > 0.8:
                    cell_data["defects"].append({"type": "Unknown_Anomaly", "conf": anom_score})

                report["failed_cells"].append(cell_data)
                
                # Aggregate confidence for final score
                max_conf = max([d['conf'] for d in defects]) if defects else anom_score
                defect_confs.append(max_conf)

        # --- STEP 7: Final PV Defective Score Calculation ---
        report["final_score"] = self._calculate_final_score(
            snr=report["snr_value"],
            total_anomalies=sum(anomaly_scores),
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

    # done
    def _calculate_snr(self, img):
        """
        Signal-to-Noise Ratio: Mean / StdDev
        High SNR = Clear Image. Low SNR = Noisy/Grainy.
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        mean, std = cv2.meanStdDev(gray)
        if std[0][0] == 0: return 0
        return float(mean[0][0] / std[0][0])

    # done
    def _filter_cell_boxes_by_width_mode(boxes, tolerance=0.05):
        """
        boxes: list of [x_min, y_min, x_max, y_max]
        tolerance: fraction deviation from mode (e.g., 0.3 → ±30%)
        """

        widths = [(int(x2)-int(x1)) for x1, y1, x2, y2 in boxes]
        if not widths:
            return 0, widths
        width_counts = Counter(widths)
        mode_width = width_counts.most_common(1)[0][0]
        

        filtered_boxes = []

        for box in boxes:
            x_min, y_min, x_max, y_max = map(int, box)
            width = x_max - x_min
            if abs(width - mode_width) / mode_width <= tolerance:
                filtered_boxes.append(box)

        return filtered_boxes
    
    # maybe done
    def _annotate_pv_by_cells(image, detections):
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

    # need to be updated
    def _extract_cells(self, img):
        """
        
        """
        CELL_DETECT_MODEL_IMGS = 1280
        CELL_PADDING = 5
        
        cells = []
        
        results = self.cell_detect_model.predict(img, conf=0.4, iou=0.6,  imgsz=CELL_DETECT_MODEL_IMGS)

        boxes = results[0].boxes.xyxy.cpu().numpy()  # list of detected boxes
        l_boxes = [list(b) for b in boxes]  # convert to Python list if needed

        filtered_boxes = self._filter_cell_boxes_by_width_mode(l_boxes)

        # sorted_boxes = self.sort_boxes_top_down_left_right(filtered_boxes)
        for i, det in enumerate(filtered_boxes):
            x1, y1, x2, y2 = map(int, det)
            
            # Add padding around the cell for context
            pad = CELL_PADDING
            h_resized, w_resized = img.shape[:2]
            
            # Calculate padded coordinates, clamping to image boundaries
            x1_padded = max(0, x1 - pad)
            y1_padded = max(0, y1 - pad)
            x2_padded = min(w_resized, x2 + pad)
            y2_padded = min(h_resized, y2 + pad)
            
            # Crop with padding
            cell_crop = img[y1_padded:y2_padded, x1_padded:x2_padded]

            if cell_crop.size == 0:
                continue

            cells.append({pos_in_pv, cell_crop}) # need pos_in_pv so it is presented in the report matrix with cell id . may be sorted from top left first

        annotated_pv = self._annotate_pv_by_cells(img, boxes)

        return cells, annotated_pv
    
    # need to be updated
    def _predict_anomaly(self, cell_img):
        """
        Runs the Anomalib/PatchCore model.
        Returns float 0.0 (Clean) to 1.0 (Highly Anomalous).
        """

        return self.anomaly_model.predict(cell_img)


    def _detect_defects(self, cell_img):
        """
        Runs YOLO on the single cell.
        """
        results = self.cell_defect_model(cell_img, verbose=False, conf=0.25)
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
        The '+' Node in your flowchart.
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

    PV_SEG_MODEL = r"runs/detect/train/weights/best.pt"
    CELL_DETECT_MODEL = r"runs/detect/train/weights/cell_detector_best_model_8s_7_1.pt"

    CELL_ANOMALY_MODEL = r"runs/detect/train/weights/best.pt"
    CELL_DEFECT_MODEL = r"runs/detect/train/weights/best.pt"
    
    # 1. Initialize logic (Do this when server starts)
    pipeline = PVInspectionPipeline(
        pv_seg_model_path=PV_SEG_MODEL,
        cell_detect_model_path=CELL_DETECT_MODEL, 
        anomaly_model_path=CELL_ANOMALY_MODEL,
        cell_defect_model_path=CELL_DEFECT_MODEL,
    )
    
    # 2. Process an upload (Do this per request)
    result_report = pipeline.analyze_image("test_panel.jpg")
    
    # 3. Send JSON back to frontend
    print("Final JSON for Web App:", result_report)