"""
Step 1b: P-K batch sampler. SupCon needs >=2 samples per identity per batch
to have any positives at all -- this sampler guarantees that.
"""

import torch


class PKSampler(torch.utils.data.Sampler):
    """
    Yields batches of P identities x K samples each (batch size = P * K).
    `labels` should be the full list/array of identity labels for the dataset,
    in the same order as the underlying Dataset's indices.
    """

    def __init__(self, labels, P: int = 12, K: int = 4, batches_per_epoch: int = 50):
        self.P, self.K = P, K
        self.batches_per_epoch = batches_per_epoch

        self.index_by_label = {}
        for idx, lab in enumerate(labels):
            self.index_by_label.setdefault(int(lab), []).append(idx)

        self.all_labels = list(self.index_by_label.keys())
        if len(self.all_labels) < P:
            raise ValueError(
                f"P={P} identities requested per batch, but only "
                f"{len(self.all_labels)} identities exist in this split."
            )

    def __iter__(self):
        for _ in range(self.batches_per_epoch):
            batch = []
            chosen = torch.randperm(len(self.all_labels))[: self.P]
            for li in chosen:
                pool = self.index_by_label[self.all_labels[li]]
                if len(pool) >= self.K:
                    picks = torch.randperm(len(pool))[: self.K]
                else:
                    # fewer than K images for this identity -> sample with replacement
                    picks = torch.randint(len(pool), (self.K,))
                batch.extend(pool[i] for i in picks.tolist())
            yield batch

    def __len__(self):
        return self.batches_per_epoch
