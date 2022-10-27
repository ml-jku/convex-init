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

    def __call__(self, weight):
        return torch.exp(weight)

    def init_raw_(self, weight, bias):
        import math
        fan_in = nn.init._calculate_correct_fan(weight, "fan_in")
        pi = math.pi
        tmp = fan_in * (2 * pi + 3 * 3 ** .5 - 6)
        mean = math.log(6 * pi) - math.log(
            fan_in * (4 * pi - 3 * 3 ** .5 + tmp) * (10 * pi - 3 * 3 ** .5 + tmp)
        ) / 2
        var = math.log(10 - 3 * 3 ** .5 + tmp) - math.log(6 * pi)
        shift = (3 * fan_in / (4 * pi - 3 * 3 ** .5 + tmp)) ** .5

        nn.init.normal_(weight, mean, var ** .5)
        nn.init.constant_(bias, -shift)


class ClippedPositivity(Positivity):

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


class ConvexLinear(nn.Linear):

    def __init__(self, *args, positivity=None, **kwargs):
        if positivity is None:
            raise TypeError("positivity must be given for convex layer")

        self.positivity = positivity
        super().__init__(*args, **kwargs)

    def forward(self, x):
        return torch.nn.functional.linear(x, self.positivity(self.weight), self.bias)

    def reset_parameters(self):
        self.positivity.init_raw_(self.weight, self.bias)


class ConvexConv2d(nn.Conv2d):

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
