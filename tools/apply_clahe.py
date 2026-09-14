"""
tools/apply_clahe.py
--------------------
Applies OpenCV CLAHE (Contrast Limited Adaptive Histogram Equalization)
to cropped palm vein images and saves processed outputs.
"""

import os
import cv2

BASE_DIR = r"c:\Users\riyat\OneDrive\Documents\palmvein"
INPUT_DIR = os.path.join(BASE_DIR, "data", "cropped")
OUTPUT_224_DIR = os.path.join(BASE_DIR, "data", "processed")
OUTPUT_448_DIR = os.path.join(BASE_DIR, "data", "processed_448")


def apply_clahe_preprocessing(clip_limit=2.0, tile_grid_size=(8, 8)):
    """
    Applies CLAHE to cropped palm vein images and resizes them to 224x224 and 448x448.
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)

    if not os.path.exists(INPUT_DIR):
        print(f"[ERROR] Input directory '{INPUT_DIR}' not found.")
        return

    subjects = sorted([
        d for d in os.listdir(INPUT_DIR)
        if os.path.isdir(os.path.join(INPUT_DIR, d))
    ])

    total_images = 0

    for subject in subjects:
        src_subject_dir = os.path.join(INPUT_DIR, subject)
        dst_224_dir = os.path.join(OUTPUT_224_DIR, subject)
        dst_448_dir = os.path.join(OUTPUT_448_DIR, subject)

        os.makedirs(dst_224_dir, exist_ok=True)
        os.makedirs(dst_448_dir, exist_ok=True)

        for filename in sorted(os.listdir(src_subject_dir)):
            if not filename.endswith(".png"):
                continue

            src_filepath = os.path.join(src_subject_dir, filename)

            # Read cropped image in Grayscale
            img = cv2.imread(src_filepath, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue

            # Apply CLAHE contrast enhancement
            enhanced_img = clahe.apply(img)

            # Resize to target resolutions
            img_224 = cv2.resize(enhanced_img, (224, 224), interpolation=cv2.INTER_AREA)
            img_448 = cv2.resize(enhanced_img, (448, 448), interpolation=cv2.INTER_AREA)

            # Save processed images
            cv2.imwrite(os.path.join(dst_224_dir, filename), img_224)
            cv2.imwrite(os.path.join(dst_448_dir, filename), img_448)

            total_images += 1

    print(f"Done! CLAHE applied to {total_images} images saved to processed (224) and processed_448 (448).")


if __name__ == "__main__":
    apply_clahe_preprocessing()
