"""
src/split_utils.py

Generates and persists the two split decisions needed for evaluation:

1. make_folds       - 5-fold identity-disjoint split of all subjects,
                       for open-set cross-validation.
2. pick_held_out_images - one held-out image per subject, for the
                       closed-set (4-train/1-test-per-subject) check.

Both are randomized but seeded, and both are meant to be generated ONCE
and then saved to disk (splits.json), so every later training/eval run
loads the same split instead of regenerating a new random one. This is
what makes fold results comparable across reruns, and lets the 224 and
448 pipelines be evaluated on identical splits.

Usage (run this file directly, once, to generate splits.json):
    python split_utils.py
"""

import json
import os
import random

SEED = 42
K_FOLDS = 5
SPLITS_FILE = os.path.join(os.path.dirname(__file__), "splits.json")


def discover_subjects(root: str):
    """List subject folder names under root, sorted for determinism."""
    return sorted([
        d for d in os.listdir(root)
        if os.path.isdir(os.path.join(root, d))
    ])


def make_folds(subject_list, k: int = K_FOLDS, seed: int = SEED):
    """
    Shuffle subject_list with a fixed seed, split into k equal-size
    groups, and return a list of k (train_subjects, test_subjects)
    pairs. test_subjects for fold i is group i; train_subjects for
    fold i is every other group combined.

    len(subject_list) must be evenly divisible by k (60 / 5 = 12,
    fine for this dataset). If it isn't, the last group will be
    smaller than the rest.
    """
    subjects = list(subject_list)
    rng = random.Random(seed)
    rng.shuffle(subjects)

    n = len(subjects)
    group_size = n // k
    groups = [subjects[i * group_size:(i + 1) * group_size] for i in range(k)]

    # If n isn't evenly divisible, tack any remainder onto the last group.
    remainder = subjects[k * group_size:]
    if remainder:
        groups[-1].extend(remainder)

    folds = []
    for i in range(k):
        test_subjects = groups[i]
        train_subjects = [s for g_idx, g in enumerate(groups) if g_idx != i for s in g]
        folds.append({"train_subjects": train_subjects, "test_subjects": test_subjects})

    return folds


def pick_held_out_images(subject_list, root: str, seed: int = SEED):
    """
    For each subject in subject_list, randomly pick one of its own
    image filenames (found under root/subject/) to hold out for
    closed-set testing. Returns {subject_name: held_out_filename}.

    One consistent seed is used for this whole dict, but each
    subject's pick is drawn from its own independent Random instance
    (seeded from SEED + subject name) so that adding/removing a
    subject from subject_list does not shift which files get picked
    for the other, unaffected subjects.
    """
    held_out = {}
    for subject in subject_list:
        folder = os.path.join(root, subject)
        images = sorted([f for f in os.listdir(folder) if f.endswith(".png")])
        if not images:
            continue
        subject_rng = random.Random(f"{seed}-{subject}")
        held_out[subject] = subject_rng.choice(images)
    return held_out


def save_splits(folds, held_out_image, path: str = SPLITS_FILE):
    with open(path, "w") as f:
        json.dump({"folds": folds, "held_out_image": held_out_image}, f, indent=2)
    print(f"Saved splits to {path}")


def load_splits(path: str = SPLITS_FILE):
    with open(path) as f:
        data = json.load(f)
    return data["folds"], data["held_out_image"]


if __name__ == "__main__":
    # Point this at whichever processed folder you're using to discover
    # the subject list (224 and 448 have the same 60 subject names, so
    # either works, this only reads folder names, not image content).
    DATASET_ROOT = r"c:\Users\riyat\OneDrive\Documents\palmvein\data\processed"

    subjects = discover_subjects(DATASET_ROOT)
    print(f"Found {len(subjects)} subjects")

    folds = make_folds(subjects, k=K_FOLDS, seed=SEED)
    for i, fold in enumerate(folds):
        print(f"Fold {i}: {len(fold['train_subjects'])} train, {len(fold['test_subjects'])} test")

    held_out_image = pick_held_out_images(subjects, DATASET_ROOT, seed=SEED)
    print(f"Picked a held-out image for {len(held_out_image)} subjects")

    save_splits(folds, held_out_image)