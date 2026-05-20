"""
Advanced loss functions for financial prediction.

- SpearmanRankLoss: differentiable rank correlation
- DirectionLoss: binary cross-entropy on sign agreement
- CombinedLoss: weighted sum of all losses
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpearmanRankLoss(nn.Module):
    """
    Differentiable approximation of Spearman rank correlation.

    Uses soft ranking (isotonic regression relaxation) to make
    the ranking operation differentiable. Penalises disagreement
    between the predicted and true ordering.

    Implementation: sort both pred and target, compute MSE on sorted values.
    This is a simplified but effective proxy for rank correlation.
    """

    def __init__(self, temperature: float = 1.0):
        super().__init__()
        self.temperature = temperature

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred:   (B,) or (B, 1)
            target: (B,) or (B, 1)

        Returns:
            scalar loss (lower = better rank correlation)
        """
        pred = pred.view(-1)
        target = target.view(-1)

        if len(pred) < 2:
            return torch.tensor(0.0, device=pred.device)

        # Sort both by target value
        _, idx = torch.sort(target)
        pred_sorted = pred[idx]
        target_sorted = target[idx]

        # Compute MSE on sorted sequences
        # This penalises pred not being monotonic with target
        mse_sorted = F.mse_loss(pred_sorted, target_sorted)

        # Also compute on rank-normalised values
        pred_rank = _soft_rank(pred, self.temperature)
        target_rank = _soft_rank(target, self.temperature)
        rank_loss = F.mse_loss(pred_rank, target_rank)

        return 0.5 * mse_sorted + 0.5 * rank_loss


def _soft_rank(x: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """
    Soft ranking: for each pair (i,j), compute P(x_i > x_j) via sigmoid.
    Sum over j to get soft rank. Scale to [-1, 1].
    """
    n = len(x)
    if n < 2:
        return torch.zeros_like(x)

    # Pairwise differences: (n, n)
    diff = x.unsqueeze(0) - x.unsqueeze(1)  # diff[i,j] = x_i - x_j
    prob = torch.sigmoid(diff / temperature)  # P(x_i > x_j)
    soft_rank = prob.sum(dim=1)  # (n,) — how many items x_i beats
    # Normalise to [-1, 1]
    soft_rank = 2.0 * (soft_rank / (n - 1)) - 1.0
    return soft_rank


class DirectionLoss(nn.Module):
    """
    Binary cross-entropy loss for predicting the sign (direction) of returns.

    We care about getting the direction right, not just the magnitude.
    """

    def __init__(self, margin: float = 0.0):
        """
        Args:
            margin: minimum |pred| to count as a confident direction call.
                    Predictions within [-margin, margin] are treated as "no call".
        """
        super().__init__()
        self.margin = margin

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred:   (B,) or (B, 1)
            target: (B,) or (B, 1)

        Returns:
            scalar loss
        """
        pred = pred.view(-1)
        target = target.view(-1)

        # Binary labels: 1 = positive return, 0 = negative
        target_sign = (target > 0).float()

        # Only count samples with |pred| > margin
        confident = (pred.abs() > self.margin).float()

        if confident.sum() == 0:
            return torch.tensor(0.0, device=pred.device)

        # Sigmoid to get probability of positive direction
        pred_prob = torch.sigmoid(pred)

        # Weighted BCE (only confident predictions)
        bce = F.binary_cross_entropy(pred_prob, target_sign, reduction="none")
        loss = (bce * confident).sum() / confident.sum().clamp_min(1)

        return loss


class CombinedLoss(nn.Module):
    """
    Weighted combination of all loss terms:

    total = α·MSE + β·LoadBalance + γ·Rank + δ·Direction
    """

    def __init__(self,
                 mse_coef: float = 1.0,
                 load_balance_coef: float = 0.01,
                 rank_coef: float = 0.3,
                 direction_coef: float = 0.1):
        super().__init__()
        self.mse_coef = mse_coef
        self.lb_coef = load_balance_coef
        self.rank_coef = rank_coef
        self.dir_coef = direction_coef

        self.rank_loss = SpearmanRankLoss()
        self.dir_loss = DirectionLoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor,
                load_balance: torch.Tensor = None) -> tuple:
        """
        Returns:
            total_loss: scalar
            losses_dict: {"mse": ..., "rank": ..., "direction": ..., "load_balance": ...}
        """
        mse = F.mse_loss(pred.view(-1), target.view(-1))
        rank = self.rank_loss(pred, target)
        direction = self.dir_loss(pred, target)

        total = self.mse_coef * mse + self.rank_coef * rank + self.dir_coef * direction

        losses = {
            "mse": mse.item(),
            "rank": rank.item(),
            "direction": direction.item(),
        }

        if load_balance is not None:
            total = total + self.lb_coef * load_balance
            losses["load_balance"] = load_balance.item() if isinstance(load_balance, torch.Tensor) else load_balance

        losses["total"] = total.item()
        return total, losses
