# test_paths.py
import os

base = "roboflow_dataset"
print("Train images:", os.path.exists(f"{base}/train/images"))
print("Valid images:", os.path.exists(f"{base}/valid/images"))
print("data.yaml:", os.path.exists(f"{base}/data.yaml"))

if not os.path.exists(f"{base}/valid"):
    print("❌ Folder missing: roboflow_dataset/valid")