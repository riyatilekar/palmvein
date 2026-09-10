"""
crop_picker.py
--------------
Interactive crop-region picker for the palm vein dataset.

Controls:
  LEFT / RIGHT arrow  : cycle through sample subjects
  SPACE               : draw / redraw the crop box on current image
  ENTER               : confirm current crop box and show preview
  S                   : save the coordinates to crop_coords.txt
  Q / ESC             : quit
"""

import cv2
import os
import numpy as np

DATASET = r"c:\Users\riyat\OneDrive\Documents\palmvein\palm_vein_dataset"

# Pick one representative image per subject (sample_000)
subjects = sorted([d for d in os.listdir(DATASET) if os.path.isdir(os.path.join(DATASET, d))])
images   = [os.path.join(DATASET, s, "sample_002.png") for s in subjects]  # middle sample

# --------------------------------------------------------------------------

def load(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    return img

def draw_grid(img, roi, label):
    """Return a display-ready BGR image with ROI rectangle and label."""
    disp = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if roi:
        x, y, w, h = roi
        cv2.rectangle(disp, (x, y), (x + w, y + h), (0, 255, 0), 2)
    # label
    cv2.putText(disp, label, (10, 28), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (0, 220, 255), 2, cv2.LINE_AA)
    cv2.putText(disp, "SPACE=pick  ENTER=preview  ARROWS=next/prev  S=save  Q=quit",
                (10, img.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (200, 200, 200), 1, cv2.LINE_AA)
    return disp

# --------------------------------------------------------------------------

def main():
    idx = 0
    roi = None

    cv2.namedWindow("Crop Picker", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Crop Picker", 1280, 800)

    while True:
        path  = images[idx]
        img   = load(path)
        label = f"{subjects[idx]}  |  {os.path.basename(path)}  |  {img.shape[1]}x{img.shape[0]}"
        disp  = draw_grid(img, roi, label)
        cv2.imshow("Crop Picker", disp)

        key = cv2.waitKey(0) & 0xFF

        if key in (ord('q'), 27):          # Q / ESC  — quit
            break

        elif key == 83 or key == ord('d'): # RIGHT arrow or d
            idx = (idx + 1) % len(images)

        elif key == 81 or key == ord('a'): # LEFT arrow or a
            idx = (idx - 1) % len(images)

        elif key == ord(' '):              # SPACE — select ROI
            cv2.imshow("Crop Picker", disp)
            r = cv2.selectROI("Crop Picker", cv2.cvtColor(img, cv2.COLOR_GRAY2BGR),
                               fromCenter=False, showCrosshair=True)
            if r[2] > 0 and r[3] > 0:
                roi = r   # (x, y, w, h)
                print(f"Selected ROI: x={r[0]}, y={r[1]}, w={r[2]}, h={r[3]}")

        elif key == 13:                    # ENTER — preview crop
            if roi:
                x, y, w, h = roi
                cropped = img[y:y+h, x:x+w]
                cv2.imshow("Preview (any key to close)", cropped)
                cv2.waitKey(0)
                cv2.destroyWindow("Preview (any key to close)")
            else:
                print("No ROI selected yet — press SPACE to pick one.")

        elif key == ord('s'):              # S — save coords
            if roi:
                x, y, w, h = roi
                out = os.path.join(os.path.dirname(DATASET), "crop_coords.txt")
                with open(out, "w") as f:
                    f.write(f"x={x}\ny={y}\nw={w}\nh={h}\n")
                print(f"Saved crop coords to {out}")
                print(f"  x={x}, y={y}, w={w}, h={h}")
            else:
                print("Nothing to save yet.")

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
