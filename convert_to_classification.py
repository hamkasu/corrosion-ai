# convert_to_classification.py

import os
import shutil

# Paths (update these if needed)
images_dir = "corrosion-segmentation-5/images"      # Folder with .jpg
labels_dir = "corrosion-segmentation-5/labels"     # Folder with .txt
output_dir = "corrosion_data"  # Where to save classification-ready data

# Create output folders
os.makedirs(os.path.join(output_dir, "corrosion"), exist_ok=True)
os.path.join(output_dir, "no_corrosion"), exist_ok=True)

# Get all image files
image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

converted = 0
for image_file in image_files:
    # Match label file (same name, .txt)
    label_file = os.path.splitext(image_file)[0] + ".txt"
    label_path = os.path.join(labels_dir, label_file)

    # Determine class
    if os.path.exists(label_path):
        # Has label → corrosion
        dest_folder = "corrosion"
    else:
        # No label → no_corrosion
        dest_folder = "no_corrosion"

    # Copy image
    src = os.path.join(images_dir, image_file)
    dst = os.path.join(output_dir, dest_folder, image_file)
    shutil.copy(src, dst)
    converted += 1

print(f"✅ Converted {converted} images to classification format!")
print(f"📁 Ready to train: {output_dir}/")
print("Now run: python train_corrosion_model.py")