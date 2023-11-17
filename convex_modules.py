from abc import ABC, abstractmethod

import torch
from torch import nn

__all__ = [
    "Positivity", "NoPositivity", "LazyClippedPositivity", "NegExpPositivity", "ExponentialPositivity",
    "ClippedPositivity", "ConvexLinear", "ConvexConv2d", "LinearSkip", "Conv2dSkip",
]


class Positivity(ABC):
    """ Interface for function that makes weights positive. """

    @abstractmethod
    def __call__(self, weight: torch.Tensor) -> torch.Tensor:
        """ Transform raw weight to positive weight. """
        ...

    def inverse_transform(self, pos_weight: torch.Tensor) -> torch.Tensor:
        """ Transform positive weight to raw weight before transform. """
        return self.__call__(pos_weight)


class NoPositivity(Positivity):
    """
    Dummy for positivity function.

    This should make it easier to compare ICNNs to regular networks.
    """

    def __call__(self, weight):
        return weight


class LazyClippedPositivity(Positivity):
    """
    Make weights positive by clipping negative weights after each update.

    References
    ----------
    Amos et al. (2017)
        Input-Convex Neural Networks.
    """

    def __call__(self, weight):
        with torch.no_grad():
            weight.clamp_(0)

        return weight


class NegExpPositivity(Positivity):
    """
    Make weights positive by applying exponential function on negative values during forward pass.

    References
    ----------
    Sivaprasad et al. (2021)
        The Curious Case of Convex Neural Networks.
    """

    def __call__(self, weight):
        return torch.where(weight < 0, weight.exp(), weight)


class ExponentialPositivity(Positivity):
    """
    Make weights positive by applying exponential function during forward pass.
    """

    def __call__(self, weight):
        return torch.exp(weight)

    def inverse_transform(self, pos_weight):
        return torch.log(pos_weight)


class ClippedPositivity(Positivity):
    """
    Make weights positive by using applying ReLU during forward pass.
    """

    def __call__(self, weight):
        return torch.relu(weight)


class ConvexLinear(nn.Linear):
    """Linear layer with positive weights."""

    def __init__(self, *args, positivity: Positivity = None, **kwargs):
        if positivity is None:
            raise TypeError("positivity must be given as kwarg for convex layer")

        self.positivity = positivity
        super().__init__(*args, **kwargs)

    def forward(self, x):
        return torch.nn.functional.linear(x, self.positivity(self.weight), self.bias)


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


class ConvexLayerNorm(nn.LayerNorm):
    """
    LayerNorm with positive weights and tracked statistics.

    Tracking statistics is necessary to make LayerNorm a convex function during inference.
    During training this module is not a convex function.
    """

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
        raw_val = self.positivity.inverse_transform(torch.ones(1)).item()
        nn.init.constant_(self.weight, raw_val)
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
    """
    Fully-connected skip-connection with learnable parameters.

    The learnable parameters of this skip-connection must not be positive
    if they skip to any hidden layer from the input.
    This is the kind of skip-connection that is commonly used in ICNNs.
    """

    def __init__(self, in_features: int, out_features: int, residual: nn.Module):
        super().__init__()
        self.skip = nn.Linear(in_features, out_features, bias=False)
        self.residual = residual

    def forward(self, x):
        return self.skip(x) + self.residual(x)

    def reset_parameters(self):
        nn.init.kaiming_normal_(self.skip.weight)
        nn.init.zeros_(self.skip.bias)


class Conv2dSkip(nn.Module):
    """
    Convolutional skip-connection with learnable parameters.

    The learnable parameters of this skip-connection must not be positive
    if they skip to any hidden layer from the input.
    This is the kind of skip-connection that is commonly used in ICNNs.
    """

    def __init__(self, in_channels: int, out_channels: int, residual: nn.Module):
        super().__init__()
        self.skip = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.residual = residual

    def forward(self, x):
        return self.skip(x) + self.residual(x)

    def reset_parameters(self):
        nn.init.kaiming_normal_(self.skip.weight)
        nn.init.zeros_(self.skip.bias)