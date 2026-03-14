# upload new images to roboflow
from roboflow import Roboflow

rf = Roboflow(api_key="60q1gKdq4vPi4tRgd831")
project = rf.workspace("digitizo").project("eldds1400c5-cell-level")                

# project.upload(
#     dataset_path="cells_ELDDS1400c5-dataset-3",
#     annotation_format="yolo",
#     num_workers=2
# )

rf.workspace("digitizo").upload_dataset(
    dataset_path="temp for pv_croped_mixed_for_int_more-cls",
    project_name="mixed-elpv-orig-res-cell-level",
    dataset_format="yolo"
)