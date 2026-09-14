"""
src/dataset.py
"""

import os
from PIL import Image
from torch.utils.data import Dataset

DATASET_ROOT = r"c:\Users\riyat\OneDrive\Documents\palmvein\data\processed"


class PalmVeinDataset(Dataset):

    def __init__(self, root: str):
        self.root    = root
        self.samples = []   # list of (abs_path, label_int)

        subjects = sorted([
            d for d in os.listdir(root)
            if os.path.isdir(os.path.join(root, d))
        ])

        for label, subject in enumerate(subjects):
            folder = os.path.join(root, subject)
            for fname in sorted(os.listdir(folder)):
                if fname.endswith(".png"):
                    self.samples.append((os.path.join(folder, fname), label))

        self.num_subjects = len(subjects)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # IMPORTANT:
        # __getitem__ should return a tensor, NOT an image.
        # This is because the dataloader expects a tensor,
        # that code then will have to be written in training 
        # script.
        #                                           -Anay

        path, label = self.samples[idx]
        img = Image.open(path)
        return img, label


if __name__ == "__main__":
    ds = PalmVeinDataset(DATASET_ROOT)
    print(f"Subjects : {ds.num_subjects}")
    print(f"Images   : {len(ds)}")

    img, label = ds[0]
    print(f"First image — size: {img.size}, mode: {img.mode}, label: {label}")
