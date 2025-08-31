# collect_corrections.py

import sqlite3
import shutil
import os

conn = sqlite3.connect("data.db")
cursor = conn.cursor()

cursor.execute("""
    SELECT image_path FROM inspections 
    WHERE corrected_label = 'corrosion'
""")

os.makedirs("retraining_data/corrosion", exist_ok=True)

for row in cursor.fetchall():
    filename = row[0].split("/")[-1]
    src = os.path.join("uploads", filename)
    dst = os.path.join("retraining_data/corrosion", filename)
    if os.path.exists(src):
        shutil.copy(src, dst)

conn.close()
print(f"✅ Collected {cursor.rowcount} corrected corrosion images for retraining")