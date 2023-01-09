import math

import torch
from torch import nn

from convex_modules import ConvexConv2d, ExponentialPositivity, ConvexLinear


class EmpiricalInit:
    """Empirical initialisation, inspired by LSUV init."""

    def __init__(self, target_mean: float = 0., target_var: float = 1.):
        self.target_mean = target_mean
        self.target_var = target_var

    @property
    def target_std(self):
        return self.target_var ** .5

    def __call__(self, module, x):
        return self.fix_prop(module, x)

    @torch.no_grad()
    def fix_prop(self, module: nn.Module, x: torch.Tensor):
        if isinstance(module, nn.Sequential):
            for lay in module.children():
                x = self.fix_prop(lay, x)

            return x
        elif isinstance(module, (ConvexLinear, ConvexConv2d)):
            return self.fix_prop_convex(module, x)
        elif isinstance(module, (nn.Linear, nn.Conv2d)):
            return self.fix_prop_linear(module, x)
        elif isinstance(module, (nn.LayerNorm, nn.BatchNorm2d)):
            return self.fix_prop_norm(module, x)
        elif sum(par.numel() for par in module.parameters()) == 0:
            return module(x)
        else:
            raise ValueError(f"no init for '{type(module)}' module")

    @torch.no_grad()
    def fix_prop_linear(self, module: nn.Module, x: torch.Tensor):
        incoming = tuple(range(1, module.weight.ndim))
        module.weight -= module.weight.mean(dim=incoming, keepdims=True)
        if module.bias is not None:
            nn.init.constant_(module.bias, self.target_mean)
        elif self.target_mean != 0.:
            # assume next layer does normalisation
            return (x - x.mean()) * module.weight.std()

        x = module(x)

        module.weight *= self.target_std / x.std()
        return (x - x.mean()) * self.target_std / x.std() + self.target_mean

    @torch.no_grad()
    def fix_prop_norm(self, module: nn.Module, x: torch.Tensor):
        nn.init.constant_(module.weight, self.target_std)
        nn.init.constant_(module.bias, self.target_mean)
        return module(x)

    @torch.no_grad()
    def fix_prop_convex(self, module: nn.Module, x: torch.Tensor):
        incoming = tuple(range(1, module.weight.ndim))
        module.weight -= module.weight.mean(dim=incoming, keepdims=True)

        x = module(x)

        module.weight += (math.log(self.target_var) - torch.log(x.var())) / 2.
        if module.bias is not None:
            module.bias += self.target_mean - x.mean()
            module.bias *= (self.target_var / x.var()) ** .5
        else:
            # assume next layer does normalisation
            return x * self.target_var / x.std()

        return (x - x.mean()) * self.target_std / x.std() + self.target_mean


if __name__ == "__main__":
    lsuv_init = EmpiricalInit()
    model = nn.Sequential(
        nn.Conv2d(1, 6, 5, stride=2, padding=2),
        nn.Sequential(nn.ReLU(), nn.Conv2d(6, 16, 5, stride=2)),
        nn.Sequential(nn.ReLU(), nn.Conv2d(16, 120, 5)),
        nn.Flatten(),
        nn.Sequential(nn.ReLU(), nn.Linear(120, 84)),
        nn.Sequential(nn.ReLU(), nn.Linear(84, 10)),
    )

    print("-" * 30)
    for name, par in model.named_parameters():
        print(
            f"{name:16s} {par.mean().item(): 6.3f}, {par.var().item():5.3f}",
            tuple(par.shape)
        )
    print("-" * 30)
    noise = lsuv_init(model, torch.randn(1024, 1, 28, 28))
    print(noise.mean(), noise.var())
    out = model(torch.randn(1024, 1, 28, 28))
    print(out.mean(), out.var())
    print("-" * 30)
    for name, par in model.named_parameters():
        print(
            f"{name:16s} {par.mean().item(): 6.3f}, {par.var().item():5.3f}",
            tuple(par.shape)
        )
    print("-" * 30)

    print("=" * 30)

    model = nn.Sequential(
        nn.Conv2d(1, 6, 5, stride=2, padding=2),
        nn.Sequential(nn.ReLU(), ConvexConv2d(6, 16, 5, stride=2, positivity=ExponentialPositivity())),
        nn.Sequential(nn.ReLU(), ConvexConv2d(16, 120, 5, positivity=ExponentialPositivity())),
        nn.Flatten(),
        nn.Sequential(nn.ReLU(), ConvexLinear(120, 84, positivity=ExponentialPositivity())),
        nn.Sequential(nn.ReLU(), ConvexLinear(84, 10, positivity=ExponentialPositivity())),
    )

    print("-" * 30)
    for name, par in model.named_parameters():
        print(
            f"{name:16s} {par.mean().item(): 6.3f}, {par.var().item():5.3f}",
            tuple(par.shape)
        )
    print("-" * 30)
    noise = lsuv_init(model, torch.randn(1024, 1, 28, 28))
    print(noise.mean(), noise.var())
    out = model(torch.randn(1024, 1, 28, 28))
    print(out.mean(), out.var())
    print("-" * 30)
    for name, par in model.named_parameters():
        print(f"{name:16s} {par.mean().item(): 6.3f}, {par.var().item():5.3f}")
    print("-" * 30)

