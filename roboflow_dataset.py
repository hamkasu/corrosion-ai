# download_from_roboflow.py
from roboflow import Roboflow

rf = Roboflow(api_key="UEVy3RH1ekFLVJYMztXn")  # Get from Roboflow profile
project = rf.workspace("hamka-corrosion").project("corrosion-segmentation-rlb4u")
dataset = project.version(5).download("yolov8")

print(f"Dataset saved to {dataset.location}")