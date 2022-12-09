import torch
from torch import nn
from torch.utils.data import DataLoader


class Whiten:

    def __init__(self, ds, keep: int = None, zca: bool = False, eps: float = 1e-5):
        x_raw, _ = next(iter(DataLoader(ds, batch_size=len(ds))))
        x_raw = torch.flatten(x_raw, start_dim=1)
        mean = x_raw.mean(dim=0)
        x_c = x_raw - mean
        cov = x_c.T @ x_c / len(x_raw)
        l, u = torch.linalg.eigh(cov)
        if keep is not None:
            l, u = l[-keep:], u[:, -keep:]

        rot = u / (l + eps) ** .5
        if zca:
            rot = rot @ u.T

        self.is_full = zca or keep is None
        self.w = rot.T
        self.b = -mean @ rot

    def __call__(self, sample):
        _sample = torch.flatten(sample, start_dim=0)
        _white = nn.functional.linear(_sample, self.w, self.b)
        return _white.view(sample.shape) if self.is_full else _white