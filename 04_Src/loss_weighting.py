"""Loss combination strategies for the multi-task model.

Each strategy exposes the same interface: combine(loss_species, loss_freshness) -> scalar.
UncertaintyWeighting carries learnable parameters and must be registered
as an nn.Module so its parameters join the optimizer alongside the model's.
"""
import torch
import torch.nn as nn


class EqualWeighting(nn.Module):
    """Unweighted sum of per-task losses."""

    def forward(self, loss_species: torch.Tensor, loss_freshness: torch.Tensor) -> torch.Tensor:
        return loss_species + loss_freshness


class UncertaintyWeighting(nn.Module):
    """Homoscedastic uncertainty weighting (Kendall et al., CVPR 2018).

    L_total = exp(-s1) * L_species + exp(-s2) * L_freshness + s1 + s2

    s1, s2 are learnable log-variances (log sigma^2), initialized to 0.
    The log-variance parameterization keeps training numerically stable
    and guarantees a positive variance.
    """

    def __init__(self):
        super().__init__()
        self.log_var_species = nn.Parameter(torch.zeros(1))
        self.log_var_freshness = nn.Parameter(torch.zeros(1))

    def forward(self, loss_species: torch.Tensor, loss_freshness: torch.Tensor) -> torch.Tensor:
        precision_species = torch.exp(-self.log_var_species)
        precision_freshness = torch.exp(-self.log_var_freshness)
        total = (
            precision_species * loss_species
            + precision_freshness * loss_freshness
            + self.log_var_species
            + self.log_var_freshness
        )
        return total.squeeze()


class DynamicWeightAveraging(nn.Module):
    """Dynamic Weight Averaging (Liu et al., CVPR 2019).

    Weights tasks by their relative rate of loss decrease over the last
    two epochs. Requires `update_epoch_losses` to be called once per
    epoch with that epoch's mean per-task training loss; weights are
    uniform until two epochs of history are available.
    """

    def __init__(self, temperature: float = 2.0):
        super().__init__()
        self.temperature = temperature
        self.num_tasks = 2
        self._history: list[tuple[float, float]] = []  # oldest first, max length 2
        self._weights = torch.ones(self.num_tasks)

    def update_epoch_losses(self, avg_loss_species: float, avg_loss_freshness: float) -> None:
        self._history.append((avg_loss_species, avg_loss_freshness))
        if len(self._history) > 2:
            self._history.pop(0)

        if len(self._history) < 2:
            self._weights = torch.ones(self.num_tasks)
            return

        (l_prev2_s, l_prev2_f), (l_prev1_s, l_prev1_f) = self._history
        r_species = l_prev1_s / l_prev2_s
        r_freshness = l_prev1_f / l_prev2_f
        r = torch.tensor([r_species, r_freshness]) / self.temperature
        self._weights = self.num_tasks * torch.softmax(r, dim=0)

    def forward(self, loss_species: torch.Tensor, loss_freshness: torch.Tensor) -> torch.Tensor:
        w_species, w_freshness = self._weights.to(loss_species.device)
        return w_species * loss_species + w_freshness * loss_freshness


def build_loss_strategy(name: str) -> nn.Module:
    strategies = {
        "EW": EqualWeighting,
        "UW": UncertaintyWeighting,
        "DWA": DynamicWeightAveraging,
    }
    if name not in strategies:
        raise ValueError(f"unknown loss weighting strategy: {name!r}, choose from {list(strategies)}")
    return strategies[name]()
