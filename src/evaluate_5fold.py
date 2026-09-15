"""
5-fold subject-disjoint evaluation of SupCon on the real palm vein dataset.

Trains a FRESH model per fold (never reuses weights across folds -- each
fold must never have seen its own test subjects), evaluates open-set
genuine/impostor separation on that fold's 12 held-out subjects, and prints
a final summary across all 5 folds: mean +/- std EER, mean +/- std d',
and a plain-language read on what the numbers mean.

Run from inside src/:
    python evaluate_5fold.py
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

PROCESSED_ROOT = os.path.join(os.path.dirname(__file__), "..", "data", "processed")

# Reduced EPOCHS 40 -> 15 based on fold 0 evidence: train EER hit a 0% floor
# by epoch 20 while test EER plateaued. DROPOUT_P adds regularization to the
# head, matching your triplet-loss notebook's EmbeddingNet (which uses
# Dropout(0.5) for the same reason -- ~240 images overfits fast otherwise).
P, K = 12, 4
BATCHES_PER_EPOCH = 30
EPOCHS = 15
LR = 1e-4
TEMPERATURE = 0.1
SEED = 42
DROPOUT_P = 0.5


def freeze_all_but_layer4(model):
    for param in model.model.parameters():
        param.requires_grad = False
    for param in model.model.layer4.parameters():
        param.requires_grad = True
    for param in model.model.fc.parameters():
        param.requires_grad = True


def add_dropout_head(model, p=0.5, embed_dim=32):
    model.model.fc = nn.Sequential(nn.Dropout(p=p), nn.Linear(512, embed_dim))
    for param in model.model.fc.parameters():
        param.requires_grad = True


def set_train_mode_frozen_bn(model):
    model.train()
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
            m.eval()


def evaluate_open_set(model, dataset):
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
    gen = sim[labels.eq(labels.t()) & ~eye]
    imp = sim[~labels.eq(labels.t())]

    d_prime = (gen.mean() - imp.mean()).abs() / torch.sqrt((gen.var() + imp.var()) / 2 + 1e-12)

    thresholds = torch.linspace(-1, 1, 2000)
    fars = torch.tensor([(imp >= t).float().mean().item() for t in thresholds])
    frrs = torch.tensor([(gen < t).float().mean().item() for t in thresholds])
    crossing = (fars - frrs).abs().argmin()
    eer = ((fars[crossing] + frrs[crossing]) / 2).item()

    # FRR at a strict fixed FAR -- the number that actually matters for a
    # deployed authentication threshold, not just the EER crossing point.
    far_target = 0.01  # 1%
    idx = (fars - far_target).abs().argmin()
    frr_at_1pct_far = frrs[idx].item()

    # "0 false positives" threshold: set right at the single highest-scoring
    # impostor pair actually observed in this fold's test data. Zero of the
    # tested impostor pairs cross this line, by construction. What matters
    # is what fraction of GENUINE pairs also get rejected at that same
    # strict line -- that's the real cost of a zero-observed-FAR target.
    zero_far_threshold = imp.max().item()
    tar_at_zero_far = (gen >= zero_far_threshold).float().mean().item()
    n_impostor_pairs = imp.numel()

    model.train()
    return {
        "d_prime": d_prime.item(),
        "eer": eer,
        "frr_at_1pct_far": frr_at_1pct_far,
        "gen_mean": gen.mean().item(),
        "imp_mean": imp.mean().item(),
        "tar_at_zero_far": tar_at_zero_far,
        "zero_far_threshold": zero_far_threshold,
        "n_impostor_pairs": n_impostor_pairs,
    }


def train_one_fold(fold_idx, train_subjects, test_subjects):
    train_ds = PalmVeinDataset(root=PROCESSED_ROOT, included_subjects=train_subjects, mode="train")
    test_ds = PalmVeinDataset(root=PROCESSED_ROOT, included_subjects=test_subjects, mode="open_set_test")
    train_labels = [lab for _, lab in train_ds.samples]

    model = ResNet18_32()
    freeze_all_but_layer4(model)
    add_dropout_head(model, p=DROPOUT_P)
    set_train_mode_frozen_bn(model)

    criterion = SupConLoss(temperature=TEMPERATURE)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)
    sampler = PKSampler(train_labels, P=min(P, len(train_subjects)), K=K,
                         batches_per_epoch=BATCHES_PER_EPOCH)

    print(f"\n--- Fold {fold_idx}: {len(train_subjects)} train / {len(test_subjects)} test subjects ---")
    t0 = time.time()
    for epoch in range(EPOCHS):
        epoch_losses = []
        for batch_idx in sampler:
            imgs = torch.stack([train_ds[i][0] for i in batch_idx])
            labels = torch.tensor([train_ds[i][1] for i in batch_idx])
            embeddings = model(imgs)
            loss = criterion(embeddings, labels)
            assert torch.isfinite(loss), f"fold {fold_idx} epoch {epoch}: loss is NaN/Inf"
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())
        if epoch % 3 == 0 or epoch == EPOCHS - 1:
            print(f"  epoch {epoch:3d}  loss={sum(epoch_losses)/len(epoch_losses):.4f}  "
                  f"elapsed={time.time()-t0:.0f}s")

    metrics = evaluate_open_set(model, test_ds)
    acc_at_eer = 100 * (1 - metrics["eer"])
    acc_at_1pct_far = 100 * (1 - metrics["frr_at_1pct_far"])
    tar_zero_far_pct = metrics["tar_at_zero_far"] * 100
    print(f"  fold {fold_idx} result: EER={metrics['eer']*100:.2f}%  "
          f"(accuracy@EER={acc_at_eer:.2f}%)  "
          f"FRR@1%FAR={metrics['frr_at_1pct_far']*100:.2f}%  "
          f"(accuracy@1%FAR={acc_at_1pct_far:.2f}%)  d'={metrics['d_prime']:.3f}")
    print(f"  fold {fold_idx} 0-observed-FAR: threshold={metrics['zero_far_threshold']:.3f} "
          f"(tested against {metrics['n_impostor_pairs']} impostor pairs)  "
          f"genuine acceptance at that threshold={tar_zero_far_pct:.2f}%")
    return metrics


def main():
    subjects = discover_subjects(PROCESSED_ROOT)
    folds = make_folds(subjects, k=5, seed=SEED)
    print(f"Discovered {len(subjects)} subjects, {len(folds)} folds")

    all_metrics = []
    for i, fold in enumerate(folds):
        m = train_one_fold(i, fold["train_subjects"], fold["test_subjects"])
        all_metrics.append(m)

    eers = torch.tensor([m["eer"] for m in all_metrics])
    frrs = torch.tensor([m["frr_at_1pct_far"] for m in all_metrics])
    dprimes = torch.tensor([m["d_prime"] for m in all_metrics])
    tars_zero_far = torch.tensor([m["tar_at_zero_far"] for m in all_metrics])
    acc_at_eer = (1 - eers) * 100
    acc_at_1pct_far = (1 - frrs) * 100
    tars_zero_far_pct = tars_zero_far * 100

    print("\n" + "=" * 78)
    print("SUMMARY -- 5-fold subject-disjoint, open-set evaluation")
    print("=" * 78)
    print(f"{'Fold':<6}{'EER %':<10}{'Acc@EER %':<12}{'FRR@1%FAR %':<14}{'Acc@1%FAR %':<14}"
          f"{'d’':<8}{'GenAcc@0FAR %':<15}")
    for i, m in enumerate(all_metrics):
        print(f"{i:<6}{m['eer']*100:<10.2f}{100*(1-m['eer']):<12.2f}"
              f"{m['frr_at_1pct_far']*100:<14.2f}{100*(1-m['frr_at_1pct_far']):<14.2f}"
              f"{m['d_prime']:<8.3f}{m['tar_at_zero_far']*100:<15.2f}")
    print("-" * 78)
    print(f"{'mean':<6}{eers.mean()*100:<10.2f}{acc_at_eer.mean():<12.2f}"
          f"{frrs.mean()*100:<14.2f}{acc_at_1pct_far.mean():<14.2f}"
          f"{dprimes.mean():<8.3f}{tars_zero_far_pct.mean():<15.2f}")
    print(f"{'std':<6}{eers.std()*100:<10.2f}{acc_at_eer.std():<12.2f}"
          f"{frrs.std()*100:<14.2f}{acc_at_1pct_far.std():<14.2f}"
          f"{dprimes.std():<8.3f}{tars_zero_far_pct.std():<15.2f}")
    print("=" * 78)

    mean_eer = eers.mean().item() * 100
    print("\nWhat this means:")
    print(f"  Mean EER across 5 subject-disjoint folds: {mean_eer:.2f}% (+/- {eers.std().item()*100:.2f}%)")
    print(f"  Equivalent accuracy at the EER threshold: {acc_at_eer.mean().item():.2f}%")
    print(f"  Equivalent accuracy at a strict 1% FAR threshold: {acc_at_1pct_far.mean().item():.2f}%")
    print(f"  Genuine acceptance if threshold is set to guarantee ZERO false accepts")
    print(f"  on the impostor pairs actually tested this run: {tars_zero_far_pct.mean().item():.2f}% "
          f"(+/- {tars_zero_far_pct.std().item():.2f}%)")
    print("  This last number is the real cost of a '0 false positives' requirement --")
    print("  it is NOT a guarantee of 0% FAR against future impostors, only against the")
    print("  specific impostor pairs measured in this run. A larger test set would likely")
    print("  push the required threshold even stricter, lowering this number further.")
    print("  (\"Accuracy\" isn't a single fixed property here -- it depends on where you")
    print("   set the acceptance threshold. These numbers show that tradeoff at three")
    print("   different operating points.)")
    if mean_eer < 1:
        verdict = ("Excellent by research standards, competitive with reported "
                    "commercial biometric systems -- but verify with more test "
                    "subjects before trusting this fully; 12 subjects/fold is small.")
    elif mean_eer < 5:
        verdict = ("Solid research-grade result for a small custom dataset. Usable "
                    "as a prototype; commercial deployment typically wants <1%.")
    elif mean_eer < 10:
        verdict = ("Workable as an early prototype, not yet deployment-ready. "
                    "Expect meaningful false accepts/rejects at a practical threshold.")
    else:
        verdict = ("Not yet reliable for authentication. Revisit data quality, "
                    "preprocessing/registration, augmentation, and loss hyperparameters "
                    "before adding more loss-function complexity.")
    print(f"  {verdict}")
    print("\n  Caveat: these numbers are specific to your dataset and protocol and are")
    print("  NOT directly comparable to numbers reported in other papers/products,")
    print("  which use different sensors, datasets, and evaluation protocols.")


if __name__ == "__main__":
    main()
