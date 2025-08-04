import torch
import polars as pl
from torch import nn


def calculate_squared_dijet_mass(x: torch.Tensor) -> torch.Tensor:
    """
    Calculates the squared dijet invariant mass for a tensor of four-vectors.

    Args:
        x: Each event has two four-vectors, and multiple events are expected.

    Returns:
        torch.Tensor: Output tensor, a one-dimensional tensor of masses.
    """
    x = torch.reshape(x, (-1, 2, 4))
    x = torch.transpose(x, 1, 2)
    x = torch.sum(x, dim=2)
    x = torch.square(x)
    x = x[..., 0] - x[..., 1] - x[..., 2] - x[..., 3]

    return x


def calculate_dijet_mass(
    x: torch.Tensor, accept_impossible: bool = False, penalty_factor: float = 1000.0
):
    """
    Calculates the dijet invariant mass for a tensor of four-vectors.

    Args:
        x: Each event has two four-vectors, and multiple events are expected.
        accept_impossible: Whether the mass from an impossible four-vector should be calculated or penalized for a loss
        function.
        penalty_factor: The factor by which negative squared masses are multiplied (and made positive) to assist in loss
        calculation. Should be vastly beyond the expected distribution.

    Returns:
        torch.Tensor: Output tensor, a one-dimensional tensor of masses.
    """
    x = calculate_squared_dijet_mass(x)

    if not accept_impossible:
        return torch.sqrt(x)

    x = x + torch.nn.functional.relu(-x) * penalty_factor
    x = torch.sqrt(x)
    return x
