from fpdf import FPDF
import datetime
import os
import cv2

class InspectionReportPDF(FPDF):
    def header(self):
        # Arial bold 15
        self.set_font('Arial', 'B', 16)
        # Title
        self.cell(0, 10, 'Solar Panel Inspection Report', 0, 1, 'C')
        # Line break
        self.ln(5)

    def footer(self):
        # Position at 1.5 cm from bottom
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        # Page number
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

    def chapter_title(self, title):
        self.set_font('Arial', 'B', 12)
        self.set_fill_color(200, 220, 255)  # Light blue background
        self.cell(0, 10, title, 0, 1, 'L', 1)
        self.ln(4)

    def chapter_body(self, body):
        self.set_font('Arial', '', 11)
        self.multi_cell(0, 6, body)
        self.ln()

def create_pdf_report(report_data, output_filename="inspection_report.pdf"):
    """
    Generates a PDF report based on the pipeline result.
    """
    pdf = InspectionReportPDF()
    pdf.add_page()

    # ===========================
    # 1. SUMMARY SECTION
    # ===========================
    pdf.chapter_title("Inspection Summary")
    
    # Determine Status Color
    status = report_data['status']
    if status == "PASS":
        pdf.set_text_color(0, 150, 0) # Green
        status_text = "PASSED"
    else:
        pdf.set_text_color(200, 0, 0) # Red
        status_text = "FAILED"

    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 10, f"Overall Status: {status_text}", 0, 1)
    
    pdf.set_text_color(0, 0, 0) # Reset to black
    pdf.set_font('Arial', '', 11)
    
    date_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    pdf.cell(0, 8, f"Date: {date_str}", 0, 1)
    pdf.cell(0, 8, f"Final Score: {report_data['final_score']:.2f}/100", 0, 1)
    pdf.cell(0, 8, f"SNR Value: {report_data['snr_value']:.2f}", 0, 1)
    pdf.cell(0, 8, f"Total Failed Cells: {len(report_data['failed_cells'])}", 0, 1)
    pdf.ln(5)

    # ===========================
    # 2. ANNOTATED IMAGE
    # ===========================
    pdf.chapter_title("Annotated Module View")
    
    # Save the numpy array image to a temporary file to insert into PDF
    temp_img_path = "temp_report_image.jpg"
    # report_data['annotated_image'] is the numpy array
    cv2.imwrite(temp_img_path, report_data['annotated_image'])
    
    # Calculate width to fit A4 page (approx 190mm wide max)
    pdf.image(temp_img_path, x=10, w=190)
    
    # Clean up temp file
    if os.path.exists(temp_img_path):
        os.remove(temp_img_path)
    
    pdf.ln(10) # Space after image

    # ===========================
    # 3. FAILED CELLS TABLE
    # ===========================
    if report_data['failed_cells']:
        pdf.add_page() # Start new page for table
        pdf.chapter_title("Detailed Defect Analysis")

        # Table Header
        pdf.set_font('Arial', 'B', 10)
        pdf.set_fill_color(240, 240, 240)
        
        col_widths = [20, 25, 60, 40, 35] # ID, Pos, Defects, Anom Score, Type
        headers = ["ID", "Pos", "Defect Details", "Anomaly?", "Confidence"]

        for i, header in enumerate(headers):
            pdf.cell(col_widths[i], 7, header, 1, 0, 'C', 1)
        pdf.ln()

        # Table Rows
        pdf.set_font('Arial', '', 9)
        pdf.set_fill_color(255, 255, 255)

        for cell in report_data['failed_cells']:
            # ID
            pdf.cell(col_widths[0], 6, str(cell['cell_id']), 1, 0, 'C')
            
            # Position
            pdf.cell(col_widths[1], 6, str(cell['pos']), 1, 0, 'C')
            
            # Defects
            defect_str = ""
            if cell['defects']:
                # Limit text length to prevent overflow
                d_list = [f"{d['type']}" for d in cell['defects']]
                defect_str = ", ".join(d_list)
            else:
                defect_str = "Unknown Anomaly"
            
            pdf.cell(col_widths[2], 6, defect_str[:30], 1, 0, 'L') # Truncate long text
            
            # Is Anomaly
            is_anom = "Yes" if cell['is_anomaly'] else "No"
            pdf.cell(col_widths[3], 6, is_anom, 1, 0, 'C')
            
            # Max Confidence (Approximation from defects or default)
            conf = "N/A"
            if cell['defects']:
                max_c = max([d['conf'] for d in cell['defects']])
                conf = f"{max_c:.2f}"
            pdf.cell(col_widths[4], 6, conf, 1, 0, 'C')
            
            pdf.ln()
    else:
        pdf.cell(0, 10, "No failed cells detected.", 0, 1)

    # Output the file
    pdf.output(output_filename)
    print(f"Report saved to: {output_filename}")