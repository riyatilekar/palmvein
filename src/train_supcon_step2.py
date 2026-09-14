"""
Step 2: SupCon wired into the real repo, on the real 60-subject dataset.

Uses:
  - src/dataset_224.py   (your PalmVeinDataset, train/eval transforms)
  - src/split_utils.py   (your seeded 5-fold subject-disjoint splitter)
  - src/supcon_loss.py   (plain SupCon, verified in step 1)
  - src/pk_sampler.py    (P-K batch sampler, verified in step 1)

SANDBOX NOTE: this environment cannot reach download.pytorch.org, so the
ResNet-18 backbone below is randomly initialized instead of ImageNet
pretrained. On your machine / Colab this download works normally -- swap
in the real ResNet18_32 from src/ResNet18_32.py and you'll get better
numbers than what this run shows. The point of this run is to verify the
LOSS + DATA PLUMBING is correct on your real images and real folds, not to
report a final accuracy number.
"""

import os
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18

sys.path.insert(0, os.path.dirname(__file__))
from dataset_224 import PalmVeinDataset
from split_utils import discover_subjects, make_folds
from supcon_loss import SupConLoss
from pk_sampler import PKSampler

torch.manual_seed(0)

PROCESSED_ROOT = os.path.join(os.path.dirname(__file__), "..", "data", "processed")

# ---------------------------------------------------------------------------
# 1. Real subject-disjoint fold, using your own split_utils
# ---------------------------------------------------------------------------
subjects = discover_subjects(PROCESSED_ROOT)
print(f"Discovered {len(subjects)} subjects under {PROCESSED_ROOT}")

folds = make_folds(subjects, k=5, seed=42)
fold = folds[0]
train_subjects, test_subjects = fold["train_subjects"], fold["test_subjects"]
print(f"Fold 0: {len(train_subjects)} train subjects, {len(test_subjects)} test subjects")
print(f"Test subjects (never seen in training): {test_subjects}")

# ---------------------------------------------------------------------------
# 2. Datasets -- your real PalmVeinDataset, real augmentation pipeline
# ---------------------------------------------------------------------------
train_ds = PalmVeinDataset(root=PROCESSED_ROOT, included_subjects=train_subjects, mode="train")
test_ds = PalmVeinDataset(root=PROCESSED_ROOT, included_subjects=test_subjects, mode="open_set_test")

print(f"Train images: {len(train_ds)}  ({train_ds.num_subjects} subjects)")
print(f"Test images:  {len(test_ds)}   ({test_ds.num_subjects} subjects, never trained on)")

train_labels = [lab for _, lab in train_ds.samples]

# ---------------------------------------------------------------------------
# 3. Model -- same architecture as your ResNet18_32, random-init (see note above)
# ---------------------------------------------------------------------------
class ResNet18_32_sandbox(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = resnet18(weights=None)  # would be IMAGENET1K_V1 outside this sandbox
        self.model.fc = nn.Linear(512, 32)

    def forward(self, x):
        return self.model(x)  # raw embedding -- SupConLoss normalizes internally


model = ResNet18_32_sandbox()
criterion = SupConLoss(temperature=0.1)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# ---------------------------------------------------------------------------
# 4. Helper: compute genuine/impostor score separation on a dataset, no grad
# ---------------------------------------------------------------------------
def evaluate_open_set(model, dataset, tag):
    # NOTE: at epoch 0, BatchNorm's running stats are freshly initialized and
    # unrepresentative of this data, which can make untrained embeddings look
    # artificially collapsed (near-zero variance). This is a known BatchNorm
    # cold-start artifact, not a property of the loss -- expect epoch-0 numbers
    # to be unreliable regardless of which loss is used.
    model.eval()
    loader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=False)
    all_z, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            emb = F.normalize(model(imgs), p=2, dim=1)
            all_z.append(emb)
            all_labels.append(labels)
    z = torch.cat(all_z)
    labels = torch.cat(all_labels).view(-1, 1)

    sim = z @ z.t()
    eye = torch.eye(len(labels), dtype=torch.bool)
    gen_mask = labels.eq(labels.t()) & ~eye
    imp_mask = ~labels.eq(labels.t())

    gen = sim[gen_mask]
    imp = sim[imp_mask]

    d_prime = (gen.mean() - imp.mean()).abs() / torch.sqrt((gen.var() + imp.var()) / 2 + 1e-12)

    # clean EER: sweep thresholds, find where FAR and FRR curves cross
    thresholds = torch.linspace(-1, 1, 2000)
    fars, frrs = [], []
    for t in thresholds:
        fars.append((imp >= t).float().mean().item())   # impostor wrongly accepted
        frrs.append((gen < t).float().mean().item())     # genuine wrongly rejected
    fars, frrs = torch.tensor(fars), torch.tensor(frrs)
    crossing = (fars - frrs).abs().argmin()
    eer = ((fars[crossing] + frrs[crossing]) / 2).item()
    eer_threshold = thresholds[crossing].item()

    print(f"  [{tag}] genuine sim:  mean={gen.mean():.4f} std={gen.std():.4f}  (n={gen.numel()})")
    print(f"  [{tag}] impostor sim: mean={imp.mean():.4f} std={imp.std():.4f}  (n={imp.numel()})")
    print(f"  [{tag}] d' = {d_prime.item():.3f}   EER = {eer*100:.1f}% @ threshold {eer_threshold:.3f}")
    model.train()
    return d_prime.item(), eer, gen, imp


# ---------------------------------------------------------------------------
# 5. Baseline: evaluate BEFORE any training (random-init features)
# ---------------------------------------------------------------------------
print("\n=== BEFORE training (random-init backbone) ===")
d0, eer0, gen0, imp0 = evaluate_open_set(model, test_ds, "open-set, epoch 0")

# ---------------------------------------------------------------------------
# 6. Train
# ---------------------------------------------------------------------------
P, K = 8, 4
BATCHES_PER_EPOCH = 6
EPOCHS = 4

sampler = PKSampler(train_labels, P=P, K=K, batches_per_epoch=BATCHES_PER_EPOCH)

print(f"\n=== Training: P={P} K={K} (batch={P*K}), {BATCHES_PER_EPOCH} batches/epoch, {EPOCHS} epochs ===")
t_start = time.time()

for epoch in range(EPOCHS):
    epoch_losses = []
    for batch_idx in sampler:
        imgs = torch.stack([train_ds[i][0] for i in batch_idx])
        labels = torch.tensor([train_ds[i][1] for i in batch_idx])

        embeddings = model(imgs)
        loss = criterion(embeddings, labels)
        assert torch.isfinite(loss), "loss is NaN/Inf"

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_losses.append(loss.item())

    print(f"epoch {epoch}  loss={sum(epoch_losses)/len(epoch_losses):.4f}  "
          f"elapsed={time.time()-t_start:.0f}s")

# ---------------------------------------------------------------------------
# 7. Evaluate AFTER training, on the SAME held-out (open-set) subjects
# ---------------------------------------------------------------------------
print("\n=== AFTER training ===")
d1, eer1, gen1, imp1 = evaluate_open_set(model, test_ds, "open-set, after training")

print("\n--- summary ---")
print(f"d'  before -> after: {d0:.3f} -> {d1:.3f}")
print(f"EER before -> after: {eer0*100:.1f}% -> {eer1*100:.1f}%")

torch.save(
    {"gen0": gen0, "imp0": imp0, "gen1": gen1, "imp1": imp1},
    os.path.join(os.path.dirname(__file__), "score_distributions.pt"),
)
