import copy
from abc import ABC, abstractmethod

import torch
from torch import nn

__all__ = [
    "Positivity", "ExponentialPositivity", "ClippedPositivity",
    "ConvexLinear", "ConvexConv2d",
    "LinearSkip", "Conv2dSkip",
]


class Positivity(ABC):

    @abstractmethod
    def __call__(self, weight: torch.Tensor) -> torch.Tensor: ...

    @abstractmethod
    def init_raw_(self, weight: nn.Parameter, bias: nn.Parameter): ...


class ExponentialPositivity(Positivity):
    """
    Make weights positive by using exponential function.
    Initialisation should be perfect for fully-connected convex layers.
    """

    def __call__(self, weight):
        return torch.exp(weight)

    def init_raw_(self, weight, bias):
        import math
        fan_in = nn.init._calculate_correct_fan(weight, "fan_in")
        pi = math.pi
        tmp = fan_in * (2 * pi + 3 * 3 ** .5 - 6) - 3 * 3 ** .5
        mean = math.log(6 * pi) - math.log(
            fan_in * (4 * pi + tmp) * (10 * pi + tmp)
        ) / 2
        var = math.log(10 + tmp) - math.log(6 * pi)

        nn.init.normal_(weight, mean, var ** .5)
        if bias is not None:
            shift = (3 * fan_in / (4 * pi + tmp)) ** .5  # fan-in * pos_mean / (2 * pi) ** .5
            nn.init.constant_(bias, -shift)


class ClippedPositivity(Positivity):
    """
    Make weights positive by using clipping.
    Initialisation stems from naive derivation and does not work that well.
    """

    def __call__(self, weight):
        return torch.relu(weight)

    def init_raw_(self, weight, bias):
        nn.init.kaiming_uniform_(weight)
        nn.init.zeros_(bias)

        # modification
        fan_in = nn.init._calculate_correct_fan(weight, "fan_in")
        with torch.no_grad():
            weight *= 4 / (8 - 3 / torch.pi) ** .5
            bias -= (3 * fan_in / (8 * torch.pi - 3)) ** .5


class NegExpPositivity(Positivity):
    """
    Make weights positive by calling the exponential function on the negative part.
    Initialisation is arbitrary, but this does not work and has not been tested extensively.
    """

    def __call__(self, weight):
        return torch.where(weight < 0, weight.exp(), weight)

    def init_raw_(self, weight, bias):
        nn.init.kaiming_normal_(weight)
        if bias is not None:
            nn.init.zeros_(bias)


class ConvexLinear(nn.Linear):
    """Linear layer with positive weights."""

    def __init__(self, *args, positivity=None, **kwargs):
        if positivity is None:
            raise TypeError("positivity must be given as kwarg for convex layer")

        self.positivity = positivity
        super().__init__(*args, **kwargs)

    def forward(self, x):
        return torch.nn.functional.linear(x, self.positivity(self.weight), self.bias)

    def reset_parameters(self):
        self.positivity.init_raw_(self.weight, self.bias)


class ConvexConv2d(nn.Conv2d):
    """Convolutional layer with positive weights."""

    def __init__(self, *args, positivity=None, **kwargs):
        if positivity is None:
            raise TypeError("positivity must be given for convex layer")

        self.positivity = positivity
        super().__init__(*args, **kwargs)

    def forward(self, x):
        return torch.nn.functional.conv2d(
            x, self.positivity(self.weight), self.bias,
            self.stride, self.padding, self.dilation, self.groups
        )

    def reset_parameters(self):
        self.positivity.init_raw_(self.weight, self.bias)


class ConvexLayerNorm(nn.LayerNorm):
    """LayerNorm with positive weights and tracked statistics."""

    def __init__(self, normalized_shape, positivity: Positivity = None,
                 eps=1e-5, affine=True, device=None, dtype=None,
                 momentum: float = 0.1, track_running_stats: bool = True):
        if positivity is None:
            raise TypeError("positivity must be given for convex layer")

        self.track_running_stats = False
        self.momentum = momentum
        super().__init__(normalized_shape, eps, affine, device, dtype)

        self.positivity = positivity
        self.track_running_stats = track_running_stats
        if self.track_running_stats:
            self.register_buffer("running_mean", torch.zeros(normalized_shape))
            self.register_buffer("running_var", torch.ones(normalized_shape))
            self.num_batches_tracked = 0
        else:
            self.register_buffer("running_mean", None)
            self.register_buffer("running_var", None)
            self.num_batches_tracked = None

        self.reset_running_stats()

    def reset_running_stats(self):
        if self.track_running_stats:
            nn.init.zeros_(self.running_mean)
            nn.init.ones_(self.running_var)
            self.num_batches_tracked = 0

    def reset_parameters(self):
        self.reset_running_stats()
        nn.init.zeros_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x):
        pos_weight = self.positivity(self.weight)
        if not self.track_running_stats:
            return nn.functional.layer_norm(
                x, self.normalized_shape, pos_weight, self.bias, self.eps
            )

        if self.training:
            dims = tuple(range(-len(self.normalized_shape), 0))
            mean = x.mean(dims, keepdim=True)
            var = x.var(dims, unbiased=False, keepdim=True)
            if self.training:
                self.num_batches_tracked += 1
                self.running_mean = (
                        (1 - self.momentum) * self.running_mean
                        + self.momentum * torch.mean(mean)
                )
                self.running_var = (
                    (1 - self.momentum) * self.running_var
                    + self.momentum * torch.mean(var)
                )
        else:
            mean, var = self.running_mean, self.running_var

        x_norm = (x - mean) / (var + self.eps) ** .5
        return pos_weight * x_norm + self.bias


class LinearSkip(nn.Module):

    def __init__(self, in_features: int, out_features: int, residual: nn.Module):
        super().__init__()
        self.skip = nn.Linear(in_features, out_features, bias=False)
        self.residual = residual

    def forward(self, x):
        return self.skip(x) + self.residual(x)

    def reset_parameters(self):
        self.skip.reset_parameters()


class Conv2dSkip(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, residual: nn.Module):
        super().__init__()
        self.skip = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.residual = residual

    def forward(self, x):
        return self.skip(x) + self.residual(x)

    def reset_parameters(self):
        nn.init.kaiming_normal_(self.skip.weight)


class BiConvex(nn.Module):

    def __init__(self, conv_net: nn.Module):
        super().__init__()
        self.conv_net = conv_net
        self.conc_net = copy.deepcopy(conv_net)

    def forward(self, *args, **kwargs):
        return self.conv_net(*args, **kwargs) - self.conc_net(*args, **kwargs)
