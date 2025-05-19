import rawpy
import imageio.v2 as imageio
import cv2
import os
import csv
from glob import glob

# Percorsi
RAW_DIR = "dataset/raw"
INPUT_DIR = "dataset/input"
TARGET_DIR = "dataset/target"
CSV_PATH = "dataset/dataset.csv"

os.makedirs(INPUT_DIR, exist_ok=True)
os.makedirs(TARGET_DIR, exist_ok=True)

BRIGHTNESS_BETA = 40
csv_rows = [("input_path", "target_path")]

# Processa immagini RAW
for raw_path in glob(f"{RAW_DIR}/*"):
    base = os.path.splitext(os.path.basename(raw_path))[0]

    # Leggi il RAW e sviluppalo come array RGB 16-bit
    with rawpy.imread(raw_path) as raw:
        rgb = raw.postprocess(output_bps=16, no_auto_bright=True, use_camera_wb=True)

    # Salva TIFF (input)
    input_path = os.path.join(INPUT_DIR, f"{base}.tiff")
    imageio.imwrite(input_path, rgb)

    # Aumenta luminosità
    bright = cv2.convertScaleAbs(rgb, alpha=1.0, beta=BRIGHTNESS_BETA)

    # Salva TIFF (target)
    target_path = os.path.join(TARGET_DIR, f"{base}_bright.tiff")
    imageio.imwrite(target_path, bright)

    # Aggiungi al CSV
    csv_rows.append((f"dataset/input/{base}.tiff", f"dataset/target/{base}_bright.tiff"))

# Scrivi il CSV
with open(CSV_PATH, mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerows(csv_rows)

print("✅ Dataset TIFF creato correttamente da RAW!")
