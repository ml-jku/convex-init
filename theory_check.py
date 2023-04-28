import math

import torch
from matplotlib import pyplot as plt


def plot_statistics(data: torch.Tensor, ref_mean: float, ref_var: float,
                    ref_corr: float, axs=None):
    if axs is None:
        _, axs = plt.subplots(1, 2)
    ax1, ax2 = axs

    print(
        f"mean: {data.mean():.3f}+-{data.mean(0).std():.3f} ({ref_mean:.3f})",
        f"var: {data.var():.3f}+-{data.var(0).std():.3f} ({ref_var:.3f})",
        sep=", "
    )

    im_kwargs = dict(vmin=-1, vmax=1, interpolation="nearest", cmap="coolwarm")
    ax1.imshow(torch.corrcoef(data.T), **im_kwargs)
    ax1.axis("off")
    ax2.imshow((1. - ref_corr) * torch.eye(data.shape[0]) + ref_corr, **im_kwargs)
    ax2.axis("off")
    return ax1, ax2


if __name__ == "__main__":
    feats, var, rho = 512, 1., 0.
    x_dist = torch.distributions.MultivariateNormal(
        torch.zeros(feats), var * ((1. - rho) * torch.eye(feats) + rho)
    )

    x = x_dist.sample((1024, ))
    plot_statistics(x, x_dist.mean[0], var, rho)
    plt.show()

    a = torch.relu(x)
    a_mean_sq = var / (2. * math.pi)
    a_mom_x = var * ((1. - rho ** 2) ** .5 + rho * math.acos(-rho)) / (2. * math.pi)
    a_mom2 = var / 2.
    plot_statistics(a, ref_mean=a_mean_sq ** .5, ref_var=a_mom2 - a_mean_sq,
                    ref_corr=(a_mom_x - a_mean_sq) / (a_mom2 - a_mean_sq))
    plt.show()

    w = 0.5 + torch.randn(feats, feats)
    s = a @ w.T
    s_mean_sq = w.mean() ** 2 * feats ** 2 * a_mean_sq
    a_cov_sum = feats * (a_mom2 + (feats - 1) * a_mom_x - feats * a_mean_sq)
    s_mom_x = w.mean() ** 2 * a_cov_sum + s_mean_sq
    s_mom2 = w.var() * feats * a_mom2 + s_mom_x
    plot_statistics(s, ref_mean=s_mean_sq ** .5, ref_var=s_mom2 - s_mean_sq,
                    ref_corr=(s_mom_x - s_mean_sq) / (s_mom2 - s_mean_sq))
    plt.show()
