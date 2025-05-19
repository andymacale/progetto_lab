import rawpy
import imageio.v2 as imageio
import cv2
import os
import csv
from glob import glob

# Directory
BASE_DIR = os.path.dirname(__file__)
RAW_DIR = os.path.join(BASE_DIR, "dataset", "raw")
INPUT_DIR = os.path.join(BASE_DIR, "dataset", "input")
TARGET_DIR = os.path.join(BASE_DIR, "dataset", "target")
CSV_PATH = os.path.join(BASE_DIR, "dataset", "dataset.csv")

os.makedirs(INPUT_DIR, exist_ok=True)
os.makedirs(TARGET_DIR, exist_ok=True)

BRIGHTNESS_BETA = 40
csv_rows = [("input_path", "target_path")]

for raw_path in glob(f"{RAW_DIR}/*"):
    base = os.path.splitext(os.path.basename(raw_path))[0]

    with rawpy.imread(raw_path) as raw:
        rgb = raw.postprocess(output_bps=8)

    input_img_path = os.path.join(INPUT_DIR, f"{base}.jpg")
    imageio.imwrite(input_img_path, rgb)

    bright = cv2.convertScaleAbs(rgb, alpha=1.0, beta=BRIGHTNESS_BETA)
    target_img_path = os.path.join(TARGET_DIR, f"{base}_bright.jpg")
    imageio.imwrite(target_img_path, bright)

    csv_rows.append((f"dataset/input/{base}.jpg", f"dataset/target/{base}_bright.jpg"))

with open(CSV_PATH, mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerows(csv_rows)

print("✅ Dataset creato con successo!")
