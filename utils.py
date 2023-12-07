import os

import torch


def make_deterministic(seed: int):
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.manual_seed(seed)
