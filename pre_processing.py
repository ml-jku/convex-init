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


class Mixup:
    """Batch-wise mixup augmentation (adapted from timm.data.mixup)

    Args:
        alpha (float): mixup alpha value, mixup is active if > 0.
        prob (float): probability of applying mixup or cutmix per batch or element
        label_smoothing (float): apply label smoothing to the mixed target tensor
        num_classes (int): number of classes for target
    """

    def __init__(
        self, alpha=1.0, prob=1.0, label_smoothing=0.0, num_classes=1000
    ):
        if alpha <= 0:
            raise ValueError(f"alpha must be greater than 0, but was {alpha}")
        self.lambda_dist = torch.distributions.Beta(alpha, alpha)
        self.mix_prob = prob
        self.label_smoothing = label_smoothing
        self.num_classes = num_classes

    @property
    def mixup_alpha(self):
        return self.lambda_dist.concentration0

    def _params_per_batch(self):
        lam = 1.0
        if torch.rand() < self.mix_prob:
            lam = self.lambda_dist.sample()
        return lam

    def _mix_batch(self, x, lam):
        x_perm = x.roll(dims=0, shifts=1)
        x_mixed = x * lam + x_perm * (1.0 - lam)
        return x_mixed

    def _mix_target(self, target, lam=1.0):
        off_value = self.label_smoothing / self.num_classes
        on_value = 1.0 - self.label_smoothing
        y1 = torch.nn.functional.one_hot(target, self.num_classes) * on_value + off_value
        y2 = y1.roll(dims=0, shifts=1)
        return y1 * lam + y2 * (1.0 - lam)

    def __call__(self, *inputs, target=None):
        # check that all inputs have same batch size
        batch_sizes = [x.size(0) for x in inputs]
        if target is not None:
            batch_sizes.append(target.size(0))
        assert (
            len(set(batch_sizes)) == 1
        ), f"All inputs to mixup must have the same batch size, got batch sizes: {batch_sizes}."

        lam = self._params_per_batch()
        mixed_inputs = tuple(self._mix_batch(x, lam) for x in inputs)

        if target is not None:
            target = self._mix_target(target, lam)
            return *mixed_inputs, target

        return mixed_inputs

