import pyrealsense2 as rs
import numpy as np
import cv2
import os
import csv
from datetime import datetime

# ---------------- CONFIG ----------------
SUBJECT_ID   = "subject063"
DATASET_ROOT = r"C:\Users\riyat\Downloads\palmvein\palm_vein_dataset"

WIDTH, HEIGHT, FPS = 1280, 800, 30

EMITTER_ON = False

MANUAL_EXPOSURE = True
EXPOSURE_VALUE  = 7500
GAIN_VALUE      = 19
# -----------------------------------------


def setup_pipeline():
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.infrared, 1, WIDTH, HEIGHT, rs.format.y8, FPS)

    profile = pipeline.start(config)
    device = profile.get_device()
    depth_sensor = device.first_depth_sensor()

    if depth_sensor.supports(rs.option.emitter_enabled):
        depth_sensor.set_option(rs.option.emitter_enabled, 1.0 if EMITTER_ON else 0.0)

    if depth_sensor.supports(rs.option.enable_auto_exposure):
        depth_sensor.set_option(rs.option.enable_auto_exposure, 0.0 if MANUAL_EXPOSURE else 1.0)

    if MANUAL_EXPOSURE:
        if depth_sensor.supports(rs.option.exposure):
            depth_sensor.set_option(rs.option.exposure, EXPOSURE_VALUE)
        if depth_sensor.supports(rs.option.gain):
            depth_sensor.set_option(rs.option.gain, GAIN_VALUE)

    return pipeline


def get_save_dir():
    save_dir = os.path.join(DATASET_ROOT, SUBJECT_ID)
    os.makedirs(save_dir, exist_ok=True)
    return save_dir


def next_sample_index(save_dir):
    existing = [f for f in os.listdir(save_dir) if f.startswith("sample_") and f.endswith(".png")]
    if not existing:
        return 0
    indices = [int(f.replace("sample_", "").replace(".png", "")) for f in existing]
    return max(indices) + 1


def get_csv_path():
    return os.path.join(DATASET_ROOT, "manifest.csv")


def ensure_csv_header(csv_path):
    if not os.path.isfile(csv_path):
        with open(csv_path, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["subject_id", "filename", "filepath", "timestamp"])


def append_csv_row(csv_path, subject_id, filename, filepath):
    with open(csv_path, mode="a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([subject_id, filename, filepath, datetime.now().isoformat()])


def remove_last_csv_row(csv_path):
    with open(csv_path, mode="r", newline="") as f:
        rows = list(csv.reader(f))
    if len(rows) > 1:  # header + at least one data row
        rows.pop()
        with open(csv_path, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(rows)


def main():
    pipeline = setup_pipeline()
    save_dir = get_save_dir()
    sample_idx = next_sample_index(save_dir)

    csv_path = get_csv_path()
    ensure_csv_header(csv_path)

    last_filepath = None  # tracks the most recent capture, for discarding

    for _ in range(30):
        pipeline.wait_for_frames()

    print(f"Saving to: {save_dir}")
    print(f"Logging to: {csv_path}")
    print("Press 'c' to capture, 'd' to discard the last capture, 'q' to quit.")

    try:
        while True:
            frames = pipeline.wait_for_frames()
            ir_frame = frames.get_infrared_frame(1)
            if not ir_frame:
                continue

            ir_image = np.asanyarray(ir_frame.get_data())

            cv2.imshow("D435 IR - Palm Vein Capture", ir_image)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('c'):
                filename = f"sample_{sample_idx:03d}.png"
                filepath = os.path.join(save_dir, filename)
                cv2.imwrite(filepath, ir_image)
                append_csv_row(csv_path, SUBJECT_ID, filename, filepath)
                print(f"Saved {filepath}")
                last_filepath = filepath
                sample_idx += 1

            elif key == ord('d'):
                if last_filepath and os.path.isfile(last_filepath):
                    os.remove(last_filepath)
                    remove_last_csv_row(csv_path)
                    print(f"Discarded {last_filepath} (removed file and CSV row)")
                    sample_idx -= 1  # so the next capture reuses this number
                    last_filepath = None  # prevents discarding the same file twice
                else:
                    print("Nothing to discard.")

            elif key == ord('q'):
                break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()