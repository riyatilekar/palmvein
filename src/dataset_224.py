"""
src/dataset_224.py
"""

import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T

DATASET_ROOT = r"c:\Users\riyat\OneDrive\Documents\palmvein\data\processed"

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# Training transform: full augmentation pipeline (randomized every call).
train_transform = T.Compose([
    T.Grayscale(num_output_channels=3),              # grayscale → 3ch for pretrained backbone
    T.RandomRotation(degrees=180, fill=0),           # rotation ±180°
    T.RandomAffine(degrees=0, scale=(0.9, 1.0), fill=0),  # zoom 90–100%, no crop, no stretch
    T.ColorJitter(brightness=0.3, contrast=0.3),     # brightness & contrast: factor range [0.7, 1.3]
    T.ToTensor(),                                    # PIL [0,255] → float tensor [0,1]
    T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

# Eval transform: deterministic, no augmentation. Used for any test-time
# embedding extraction (open-set or closed-set), so scores are reproducible
# and not affected by random rotation/zoom/jitter.
eval_transform = T.Compose([
    T.Grayscale(num_output_channels=3),
    T.ToTensor(),
    T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

VALID_MODES = ("train", "open_set_test", "closed_set_test")


class PalmVeinDataset(Dataset):
    """
    Parameters
    ----------
    root : str
        Path to the folder containing one subfolder per subject
        (e.g. data/processed/subject001/sample_000.png, ...).
        Defaults to DATASET_ROOT.
    included_subjects : list[str] | None
        Which subject folder names to include. None = include all
        subjects found under root. Use this to build a fold-specific
        dataset (e.g. only the 48 training subjects, or only the 12
        held-out subjects for that fold) without moving any files.
    mode : str
        One of "train", "open_set_test", "closed_set_test".
        - "train": uses train_transform (augmented). If held_out_image
          is given, each subject's held-out file is EXCLUDED from
          training.
        - "open_set_test": uses eval_transform (deterministic). All
          images of included_subjects are kept (these subjects were
          never trained on, so there's nothing to exclude per-image).
        - "closed_set_test": uses eval_transform (deterministic). Only
          the single held-out file per subject (from held_out_image)
          is kept, everything else is excluded.
    held_out_image : dict[str, str] | None
        Maps subject folder name -> filename reserved for closed-set
        testing (e.g. {"subject007": "sample_003.png", ...}).
        Required for "train" (to know what to exclude) and for
        "closed_set_test" (to know what to keep). Not needed for
        "open_set_test".
    """

    def __init__(
        self,
        root: str = DATASET_ROOT,
        included_subjects=None,
        mode: str = "train",
        held_out_image=None,
    ):
        if mode not in VALID_MODES:
            raise ValueError(f"mode must be one of {VALID_MODES}, got {mode!r}")

        self.root = root
        self.mode = mode
        self.held_out_image = held_out_image or {}
        self.samples = []  # list of (abs_path, label_int)

        all_subjects = sorted([
            d for d in os.listdir(root)
            if os.path.isdir(os.path.join(root, d))
        ])

        if included_subjects is None:
            subjects = all_subjects
        else:
            included_set = set(included_subjects)
            subjects = [s for s in all_subjects if s in included_set]

        for label, subject in enumerate(subjects):
            folder = os.path.join(root, subject)
            held_out_fname = self.held_out_image.get(subject)  # None if not present

            for fname in sorted(os.listdir(folder)):
                if not fname.endswith(".png"):
                    continue

                if self.mode == "train":
                    # Exclude this subject's held-out file, if any.
                    if held_out_fname is not None and fname == held_out_fname:
                        continue
                elif self.mode == "closed_set_test":
                    # Keep ONLY this subject's held-out file.
                    if held_out_fname is None or fname != held_out_fname:
                        continue
                # mode == "open_set_test": keep everything, no per-image filtering.

                self.samples.append((os.path.join(folder, fname), label))

        self.num_subjects = len(subjects)

        # Pick which transform to apply, once, based on mode.
        self.transform = train_transform if self.mode == "train" else eval_transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path)
        img = self.transform(img)
        return img, label