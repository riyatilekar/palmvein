"""
Step 1: plain SupCon loss (Khosla et al. 2020), L_out formulation.
No margin, no weighting, no auxiliary terms yet. Get this verified first.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SupConLoss(nn.Module):
    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor):
        """
        embeddings: (B, D) raw network output, NOT yet normalized
        labels:     (B,)   identity label per sample (subject+palm-side combined)
        """
        device = embeddings.device
        B = embeddings.shape[0]

        z = F.normalize(embeddings, p=2, dim=1)          # unit hypersphere
        sim = z @ z.t()                                  # (B, B) cosine similarities

        labels = labels.contiguous().view(-1, 1)
        eye = torch.eye(B, dtype=torch.bool, device=device)
        same_label = labels.eq(labels.t())
        pos_mask = same_label & ~eye                     # positives, self excluded
        valid_mask = ~eye                                # A(i): everyone but self

        logits = sim / self.temperature
        # subtract row max for numerical stability, ignoring the masked diagonal
        row_max = logits.masked_fill(~valid_mask, float("-inf")).max(dim=1, keepdim=True).values
        logits = logits - row_max.detach()

        exp_logits = torch.exp(logits) * valid_mask
        denom = exp_logits.sum(dim=1, keepdim=True).clamp_min(1e-12)
        log_prob = logits - torch.log(denom)              # (B, B)

        pos_mask_f = pos_mask.float()
        num_pos = pos_mask_f.sum(dim=1)
        has_pos = num_pos > 0

        mean_log_prob_pos = (pos_mask_f * log_prob).sum(dim=1) / num_pos.clamp_min(1e-12)
        loss_per_anchor = -mean_log_prob_pos[has_pos]

        if loss_per_anchor.numel() == 0:
            # keeps the graph alive if a batch somehow has no positives anywhere
            return embeddings.sum() * 0.0
        return loss_per_anchor.mean()
