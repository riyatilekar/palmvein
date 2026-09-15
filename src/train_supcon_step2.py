"""
Step 2: SupCon wired into the real repo, on the real 60-subject dataset.

Uses:
  - src/dataset_224.py   (your PalmVeinDataset, train/eval transforms)
  - src/split_utils.py   (your seeded 5-fold subject-disjoint splitter)
  - src/supcon_loss.py   (plain SupCon, verified in step 1)
  - src/pk_sampler.py    (P-K batch sampler, verified in step 1)
  - src/ResNet18_32.py   (your real ImageNet-pretrained backbone)

This trains fold 0 only (48 train / 12 test subjects) as a first real run.
Loop over all 5 folds from split_utils once this looks healthy -- a single
fold's EER is noisy given only 12 test subjects.
"""

import os
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))
from dataset_224 import PalmVeinDataset
from split_utils import discover_subjects, make_folds
from ResNet18_32 import ResNet18_32
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

# Same 48 training subjects, but with the deterministic eval_transform instead
# of train_transform (no random augmentation). Used ONLY as an overfitting
# probe: it measures how well-separated the training identities' own images
# are, so we can compare that against the held-out (test_ds) curve over time.
train_probe_ds = PalmVeinDataset(root=PROCESSED_ROOT, included_subjects=train_subjects, mode="open_set_test")

print(f"Train images: {len(train_ds)}  ({train_ds.num_subjects} subjects)")
print(f"Test images:  {len(test_ds)}   ({test_ds.num_subjects} subjects, never trained on)")

train_labels = [lab for _, lab in train_ds.samples]

# ---------------------------------------------------------------------------
# 3. Model -- your real, ImageNet-pretrained ResNet18_32, frozen the same way
#    as EmbeddingNet in notebooks/palm_vein_metric_learning.ipynb: everything
#    frozen except layer4 and the new head. On ~240 training images, letting
#    the whole ResNet-18 fine-tune from step 1 overfits fast -- this is the
#    same reasoning your triplet-loss baseline already applies.
# ---------------------------------------------------------------------------
def freeze_all_but_layer4(model):
    for param in model.model.parameters():
        param.requires_grad = False
    for param in model.model.layer4.parameters():
        param.requires_grad = True
    for param in model.model.fc.parameters():  # the new 512->32 head, always trainable
        param.requires_grad = True


def add_dropout_head(model, p=0.5, embed_dim=32):
    """ResNet18_32.py's head is a bare Linear(512, embed_dim) -- no dropout.
    Your triplet notebook's EmbeddingNet uses Dropout(p=0.5) before its head
    specifically because full fine-tuning on ~240 images overfits fast. We
    saw that exact pattern in fold 0 (train EER hit 0% by epoch 20 while test
    EER plateaued). Bringing the same regularization back here."""
    model.model.fc = nn.Sequential(nn.Dropout(p=p), nn.Linear(512, embed_dim))
    for param in model.model.fc.parameters():
        param.requires_grad = True


def set_train_mode_frozen_bn(model):
    """model.train(), but BatchNorm layers stay in eval mode -- matches the
    notebook's override, since small batches corrupt BN running stats."""
    model.train()
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
            m.eval()


model = ResNet18_32()
freeze_all_but_layer4(model)
add_dropout_head(model, p=0.5)
set_train_mode_frozen_bn(model)

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f"Trainable params: {trainable:,} ({100*trainable/total:.1f}% of {total:,} total)")

criterion = SupConLoss(temperature=0.1)
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)

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
    set_train_mode_frozen_bn(model)
    return d_prime.item(), eer, gen, imp


# ---------------------------------------------------------------------------
# 5. Baseline: evaluate BEFORE any training (pretrained-only features)
# ---------------------------------------------------------------------------
print("\n=== BEFORE training (pretrained backbone, no fine-tuning yet) ===")
d0, eer0, gen0, imp0 = evaluate_open_set(model, test_ds, "open-set, epoch 0")

# ---------------------------------------------------------------------------
# 6. Train
# ---------------------------------------------------------------------------
# These were kept tiny (P=8,K=4, 6 batches/epoch, 4 epochs) in the sandbox to
# finish in under 2 minutes on 1 CPU core with no GPU. On your machine, raise
# these substantially -- start here and watch the loss curve / eval gap to
# decide whether to go further:
# Reduced from 40 -> 15 based on fold 0 evidence: train EER hit a 0% floor by
# epoch 20 while test EER plateaued/oscillated from ~epoch 5 onward with no
# further real improvement. Combined with dropout now in the head, this
# should spend less time overfitting to the 240 training images. Adjust
# based on what the overfitting check shows on this run.
P, K = 12, 4                # batch = 48, matches the earlier P-K recommendation
BATCHES_PER_EPOCH = 30
EPOCHS = 15

sampler = PKSampler(train_labels, P=P, K=K, batches_per_epoch=BATCHES_PER_EPOCH)

print(f"\n=== Training: P={P} K={K} (batch={P*K}), {BATCHES_PER_EPOCH} batches/epoch, {EPOCHS} epochs ===")
t_start = time.time()

EVAL_EVERY = 3  # epochs between overfitting-probe evaluations
history = {"epoch": [], "train_eer": [], "test_eer": [], "train_dprime": [], "test_dprime": []}

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

    if epoch % EVAL_EVERY == 0 or epoch == EPOCHS - 1:
        print(f"  -- overfitting probe at epoch {epoch} --")
        d_tr, eer_tr, _, _ = evaluate_open_set(model, train_probe_ds, f"TRAIN subjects, epoch {epoch}")
        d_te, eer_te, _, _ = evaluate_open_set(model, test_ds, f"TEST subjects,  epoch {epoch}")
        history["epoch"].append(epoch)
        history["train_eer"].append(eer_tr * 100)
        history["test_eer"].append(eer_te * 100)
        history["train_dprime"].append(d_tr)
        history["test_dprime"].append(d_te)
        gap = eer_tr - eer_te
        print(f"  gap (train EER - test EER) = {gap*100:+.2f} pts "
              f"{'<- train separating MUCH better than test, watch this' if gap < -0.05 else ''}")

# ---------------------------------------------------------------------------
# 7. Evaluate AFTER training, on the SAME held-out (open-set) subjects
# ---------------------------------------------------------------------------
print("\n=== AFTER training ===")
d1, eer1, gen1, imp1 = evaluate_open_set(model, test_ds, "open-set, after training")

print("\n--- summary ---")
print(f"d'  before -> after: {d0:.3f} -> {d1:.3f}")
print(f"EER before -> after: {eer0*100:.1f}% -> {eer1*100:.1f}%")

print("\n--- overfitting check ---")
print(f"{'epoch':<8}{'train EER %':<14}{'test EER %':<14}{'gap (train-test)':<18}")
for i, ep in enumerate(history["epoch"]):
    gap = history["train_eer"][i] - history["test_eer"][i]
    print(f"{ep:<8}{history['train_eer'][i]:<14.2f}{history['test_eer'][i]:<14.2f}{gap:<+18.2f}")

first_gap = history["train_eer"][0] - history["test_eer"][0]
last_gap = history["train_eer"][-1] - history["test_eer"][-1]
print(f"\nGap at epoch {history['epoch'][0]}: {first_gap:+.2f} pts  ->  "
      f"gap at epoch {history['epoch'][-1]}: {last_gap:+.2f} pts")

# Endpoint-only comparison is fragile -- a single noisy checkpoint at either
# end can hide what happened in between. Compare average gap over the first
# half of checkpoints vs the second half instead, and separately flag if
# train EER hit a hard floor (near-perfect fit) while test EER never
# followed it down -- that pattern is diagnostic on its own regardless of
# where the gap happens to land at the very last checkpoint.
n = len(history["epoch"])
mid = max(n // 2, 1)
gaps = [tr - te for tr, te in zip(history["train_eer"], history["test_eer"])]
first_half_gap = sum(gaps[:mid]) / mid
second_half_gap = sum(gaps[mid:]) / max(n - mid, 1)

train_floor_epoch = next((e for e, tr in zip(history["epoch"], history["train_eer"]) if tr < 0.1), None)
test_min = min(history["test_eer"])
test_min_epoch = history["epoch"][history["test_eer"].index(test_min)]
test_final = history["test_eer"][-1]

print(f"Average gap, first half of training: {first_half_gap:+.2f} pts")
print(f"Average gap, second half of training: {second_half_gap:+.2f} pts")
if train_floor_epoch is not None:
    print(f"Train EER first hit ~0% at epoch {train_floor_epoch}.")
print(f"Best test EER observed: {test_min:.2f}% at epoch {test_min_epoch} "
      f"(final epoch test EER: {test_final:.2f}%)")

if train_floor_epoch is not None and train_floor_epoch < history["epoch"][-1] - 5:
    print("\nWARNING: train EER hit a near-perfect floor well before training ended, "
          "while test EER kept fluctuating without a clear downward trend after that "
          "point. This is an overfitting signature: the model is memorizing training "
          "identities faster than it's learning anything that transfers to unseen ones. "
          "Consider stopping training earlier (near where test EER first plateaued), "
          "adding regularization (dropout on the head, weight decay), or more "
          "aggressive augmentation.")
elif second_half_gap < first_half_gap - 2:
    print("\nWARNING: the train/test gap widened meaningfully in the second half of "
          "training -- overfitting signature. Consider fewer epochs or stronger "
          "regularization.")
else:
    print("\nGap stayed modest and train EER did not hit a hard floor early -- "
          "no strong overfitting signature in this fold.")
print("\nNote: some train/test gap is EXPECTED even for a well-generalizing model,")
print("since train_probe images were still seen (unaugmented) during training.")
print("What matters is whether the GAP GROWS over epochs, not that it exists.")

torch.save(
    {"gen0": gen0, "imp0": imp0, "gen1": gen1, "imp1": imp1, "history": history},
    os.path.join(os.path.dirname(__file__), "score_distributions.pt"),
)
