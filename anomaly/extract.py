import json

with open('c:/Users/Rowan/Documents/Rowan/Yolo_test/anomaly/trying-3-anomaly-models.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

code_cells = []
for cell in nb.get('cells', []):
    if cell.get('cell_type') == 'code':
        source = cell.get('source', [])
        # source can be a list of strings or a single string
        if isinstance(source, list):
            source = ''.join(source)
        code_cells.append(source)

with open('c:/Users/Rowan/Documents/Rowan/Yolo_test/anomaly/trying-3-anomaly-models.py', 'w', encoding='utf-8') as f:
    f.write('\n\n# --- CELL ---\n\n'.join(code_cells))
print("Extraction complete")
