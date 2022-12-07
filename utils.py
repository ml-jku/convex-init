import os

import torch
from torch import nn


def make_deterministic(seed: int):
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(seed)


def lecun_init_(w: torch.Tensor, b: torch.Tensor) -> None:
    """
    Initialise weights according to (Lecun et al., 1998).

    Parameters
    ----------
    w : torch.Tensor
        Weight matrix to initialise.
    b : torch.Tensor
        Bias vector to initialise.
    """
    nn.init.kaiming_normal_(w, nonlinearity="linear")
    nn.init.zeros_(b)


def he_init_(w: torch.Tensor, b: torch.Tensor) -> None:
    """
    Initialise weights according to (He et al., 2015).

    Parameters
    ----------
    w : torch.Tensor
        Weight matrix to initialise.
    b : torch.Tensor
        Bias vector to initialise.
    """
    nn.init.kaiming_normal_(w, nonlinearity="relu")
    nn.init.zeros_(b)
