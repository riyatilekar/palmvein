"""
crop_all.py
-----------
Reads crop_coords.txt and applies that exact crop to every image
in the palm_vein_dataset, saving results under palm_vein_cropped/.

Output structure mirrors the original:
  palm_vein_cropped/
    subject001/
      sample_000.png
      ...
"""

import cv2
import os

BASE        = r"c:\Users\riyat\OneDrive\Documents\palmvein"
DATASET     = os.path.join(BASE, "palm_vein_dataset")
OUTPUT      = os.path.join(BASE, "palm_vein_cropped")
COORDS_FILE = os.path.join(BASE, "crop_coords.txt")

coords = {}
with open(COORDS_FILE) as f:
    for line in f:
        line = line.strip()
        if "=" in line:
            k, v = line.split("=")
            coords[k.strip()] = int(v.strip())

x, y, w, h = coords["x"], coords["y"], coords["w"], coords["h"]
print(f"Crop region: x={x}, y={y}, w={w}, h={h}  ->  output size: {w}x{h} px")

subjects = sorted([d for d in os.listdir(DATASET)
                   if os.path.isdir(os.path.join(DATASET, d))])

total_ok = 0
total_err = 0

for subj in subjects:
    src_dir = os.path.join(DATASET, subj)
    dst_dir = os.path.join(OUTPUT, subj)
    os.makedirs(dst_dir, exist_ok=True)

    images = sorted([f for f in os.listdir(src_dir) if f.endswith(".png")])
    for fname in images:
        src_path = os.path.join(src_dir, fname)
        dst_path = os.path.join(dst_dir, fname)

        img = cv2.imread(src_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"  [WARN] could not read {src_path}")
            total_err += 1
            continue

        ih, iw = img.shape
        x2, y2 = min(x + w, iw), min(y + h, ih)
        cropped = img[y:y2, x:x2]

        cv2.imwrite(dst_path, cropped)
        total_ok += 1

    print(f"  {subj}: {len(images)} images cropped")

print(f"\nDone! {total_ok} images saved to '{OUTPUT}'")
if total_err:
    print(f"  {total_err} images had errors.")
