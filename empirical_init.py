import math

import torch
from matplotlib import pyplot as plt
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
            nn.init.kaiming_normal_(module.weight)
            nn.init.zeros_(module.bias)
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
        module.weight -= module.weight.mean(dim=incoming, keepdim=True)
        if module.bias is not None:
            nn.init.constant_(module.bias, self.target_mean)
        elif self.target_mean != 0.:
            # assume next layer does normalisation
            return (x - x.mean()) * module.weight.std()

        x = module(x)

        x_var = torch.mean(x ** 2, dim=(0, *range(2, x.ndim)), keepdim=True)
        scale = self.target_std / x_var ** .5
        module.weight *= scale.view(-1, *(1 for _ in incoming))
        return (x - x.mean()) * scale + self.target_mean

    @torch.no_grad()
    def fix_prop_norm(self, module: nn.Module, x: torch.Tensor):
        nn.init.constant_(module.weight, self.target_std)
        nn.init.constant_(module.bias, self.target_mean)
        return module(x)

    @torch.no_grad()
    def fix_prop_convex(self, module: nn.Module, x: torch.Tensor):
        incoming = tuple(range(1, module.weight.ndim))
        module.weight -= module.weight.mean(dim=incoming, keepdim=True)

        x = module(x)

        x_mean = torch.mean(x, dim=(0, *range(2, x.ndim)), keepdim=True)
        x_var = torch.var(x, dim=(0, *range(2, x.ndim)), keepdim=True)
        scale = (self.target_var / x_var) ** .5
        module.weight += torch.log(scale.view(-1, *(1 for _ in incoming)))
        if module.bias is not None:
            module.bias += self.target_mean - x_mean.squeeze()
            module.bias *= scale.squeeze()
        else:
            # assume next layer does normalisation
            return x * scale

        return (x - x_mean) * scale + self.target_mean


def _test_empirical_init(model, inputs, axes=None):
    lsuv_init = EmpiricalInit()
    print("-" * 30)
    print("BEFORE")
    for name, par in model.named_parameters():
        print(
            f"{name:16s}",
            f"{par.mean().item(): 6.3f},",
            f"{par.var().item():6.4f}",
            tuple(par.shape),
        )
    print("-" * 30)

    if axes is not None:
        for ax, layer in zip(axes, model):
            with torch.no_grad():
                noise = lsuv_init(layer, inputs)
                noise_means, noise_vars = noise.mean(dim=0), noise.var(dim=0)
                print(f"{noise_means.mean().item(): .3f}+-{noise_means.std().item():5.3f}",
                      f"{noise_vars.mean().item(): .3f}+-{noise_vars.std().item():5.3f}")
                inputs = out = layer(inputs)
                out_means, out_vars = out.mean(dim=0), out.var(dim=0)
                out_mom2 = torch.mean(out ** 2, dim=0)
                print(f"{out_means.mean().item(): .3f}+-{out_means.std().item():5.3f}",
                      f"{out_vars.mean().item(): .3f}+-{out_vars.std().item():5.3f}",
                      f"{out_mom2.mean().item(): .3f}+-{out_mom2.std().item():5.3f}")
                out_cov = torch.cov(out.view(len(out), -1).T)
            bound = max(-out_cov.min(), out_cov.max())
            im = ax.imshow(out_cov, interpolation="nearest",
                           vmin=-bound, vmax=bound, cmap="RdBu_r")
            ax.axis("off")
            plt.colorbar(im, ax=ax)

    print("-" * 30)
    print("AFTER")
    for name, par in model.named_parameters():
        print(
            f"{name:16s}",
            f"{par.mean().item(): 6.3f},",
            f"{par.var().item():6.4f}",
        )
    print("-" * 30)

if __name__ == "__main__":
    x = torch.randn(1024, 1, 28, 28)
    base_model = nn.Sequential(
        nn.Conv2d(1, 6, 5, stride=2, padding=2),
        nn.Sequential(nn.ReLU(), nn.Conv2d(6, 16, 5, stride=2)),
        nn.Sequential(nn.ReLU(), nn.Conv2d(16, 120, 5)),
        nn.Sequential(nn.Flatten(), nn.ReLU(), nn.Linear(120, 84)),
        nn.Sequential(nn.ReLU(), nn.Linear(84, 10)),
    )
    convex_model = nn.Sequential(
        nn.Conv2d(1, 6, 5, stride=2, padding=2),
        nn.Sequential(nn.ReLU(), ConvexConv2d(6, 16, 5, stride=2, positivity=ExponentialPositivity())),
        nn.Sequential(nn.ReLU(), ConvexConv2d(16, 120, 5, positivity=ExponentialPositivity())),
        nn.Sequential(nn.Flatten(), nn.ReLU(), ConvexLinear(120, 84, positivity=ExponentialPositivity())),
        nn.Sequential(nn.ReLU(), ConvexLinear(84, 10, positivity=ExponentialPositivity())),
    )

    fig, axes = plt.subplots(len(base_model), 2, figsize=(48, 24 * len(base_model)))
    axes[0, 0].set_title("non-convex")
    _test_empirical_init(base_model, x, axes[:, 0])

    print("=" * 30)

    axes[0, 1].set_title("convex")
    _test_empirical_init(convex_model, x, axes[:, 1])
    plt.show()
