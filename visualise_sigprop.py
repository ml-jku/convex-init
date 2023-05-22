import math

import torch
from torch import nn
from matplotlib import pyplot as plt, gridspec

from convex_modules import ConvexLinear, LazyClippedPositivity, LinearSkip


def estimate_moments(data: torch.Tensor):
    mom1 = torch.mean(data, dim=0)
    mom2 = data.T @ data / len(data)
    return mom1, torch.diag(mom2), mom2


def normalised_moments(mixed: torch.Tensor, mean: torch.Tensor):
    cov = mixed - torch.outer(mean, mean)
    var = torch.diag(cov)
    std = var ** .5
    corr = cov / torch.outer(std, std)
    return var, cov, corr


def minify_matrix(cov: torch.Tensor, factor: int = 100):
    n_mini = cov.shape[0] // factor
    cov_tmp = cov.reshape(factor, n_mini, factor, n_mini)
    cov_blocks = torch.movedim(cov_tmp, 1, -1).reshape(-1, n_mini, n_mini)
    cov_mini = cov_blocks[::factor + 1].mean(0)
    return cov_mini


def plot_fancy_hist(data, bin_width: float = .5, color=None, ax=None):
    fig = plt.gcf()
    if ax is None:
        ax = fig.gca()

    low, high = data.min(), data.max()
    print(data.mean(), data.median())
    if low < bin_width and high > -bin_width:
        min_bin_count = math.ceil(2 * max(-low, high) / bin_width)
        bound = bin_width * (min_bin_count + .5)
        bin_count = min(int(min_bin_count), 500)
        print(-bound, bound, bin_count)
        ax.hist(data, range=(-bound, bound), bins=bin_count,
                density=True, color=color)
        return ax

    # split axis
    gs = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=ax, wspace=0.05)
    axl = fig.add_subplot(gs[0], sharey=ax)
    axr = fig.add_subplot(gs[1], sharey=ax)
    ax.remove()
    axl.spines.right.set_visible(False)
    axr.spines.left.set_visible(False)
    axr.yaxis.set_visible(False)
    kwargs = {"marker": [(-1, -.5), (1, .5)], "markersize": 12, "mew": 1}
    kwargs.update(linestyle="none", color="k", clip_on=False)
    axl.plot([1, 1], [0, 1], transform=axl.transAxes, **kwargs)
    axr.plot([0, 0], [0, 1], transform=axr.transAxes, **kwargs)

    if low > bin_width:
        gap_bound = bin_width * math.floor(low / bin_width)
        out_bound = bin_width * math.ceil(high / bin_width)
        bin_count = min(int((out_bound - gap_bound) // bin_width), 500)
        print(gap_bound, out_bound, bin_count)
        axl.hist([], range=(-out_bound, -gap_bound))
        axr.hist(data, range=(gap_bound, out_bound), bins=bin_count, density=True)
        return axr
    elif high < -bin_width:
        gap_bound = bin_width * math.ceil(high / bin_width)
        out_bound = bin_width * math.floor(low / bin_width)
        bin_count = min(int((gap_bound - out_bound) // bin_width), 500)
        print(gap_bound, out_bound, bin_count)
        axl.hist(data, range=(out_bound, gap_bound), bins=bin_count, density=True)
        axr.hist([], range=(-gap_bound, -out_bound))
        return axl

    return


def plot_moments(data, raw: bool = False, mini: bool = False, axes=None):
    if axes is None:
        _, axes = plt.subplots(ncols=3)

    ax1, ax2, ax3 = axes
    mean, mom2, mixed = estimate_moments(data)
    var, cov, corr = normalised_moments(mixed, mean=mean)
    mat, diag = (mixed, mom2) if raw else (cov, var)
    if mini:
        mat, corr = minify_matrix(mat), minify_matrix(corr)
        diag = torch.diag(mat)

    plot_fancy_hist(data.ravel(), bin_width=0.5, ax=ax1)

    ax2.axis("off")
    bound = diag.max()
    img = ax2.imshow(mat, vmin=-bound, vmax=bound, cmap="coolwarm")
    ax2.get_figure().colorbar(img, ax=ax2)

    ax3.axis("off")
    ax3.imshow(corr, vmin=-1, vmax=1, cmap="coolwarm")
    return axes


if __name__ == "__main__":
    x = torch.randn(1024, 200)
    net = torch.nn.Sequential(
        nn.Linear(200, 400),
        nn.Sequential(nn.ReLU(), nn.Linear(400, 300)),
        nn.Sequential(nn.ReLU(), nn.Linear(300, 400)),
        nn.Sequential(nn.ReLU(), nn.Linear(400, 100)),
    )

    # regular init
    with torch.no_grad():
        nn.init.kaiming_uniform_(net[0].weight, nonlinearity="linear")
        nn.init.zeros_(net[0].bias)
        for layer in net[1:]:
            nn.init.kaiming_uniform_(layer[1].weight, nonlinearity="relu")
            nn.init.zeros_(layer[1].bias)

    with torch.no_grad():
        s, = nn_results = [x]
        for layer in net:
            s = layer(s)
            nn_results.append(s)

    size = (3, len(nn_results) - 1)
    fig, axes = plt.subplots(*size, figsize=(size[1] * 6.4, size[0] * 4.8))
    print(axes[0, 0].get_shared_y_axes())
    axes[0, 0].get_shared_y_axes().join(*axes[0])
    print(axes[0, 0].get_shared_y_axes())
    for res, axes in zip(nn_results, axes.T):
        plot_moments(res, raw=True, mini=True, axes=axes)

    fig.savefig("propagation_nn.svg")
    fig.show()

    # icnn init
    with torch.no_grad():
        for layer in net[1:]:
            torch.relu_(layer[1].weight)

    with torch.no_grad():
        s, = icnn_results = [x]
        for layer in net:
            s = layer(s)
            icnn_results.append(s)

    size = (3, len(icnn_results) - 1)
    fig, axes = plt.subplots(*size, figsize=(size[1] * 6.4, size[0] * 4.8))
    axes[0, 0].get_shared_y_axes().join(*axes[0])
    for res, axes in zip(icnn_results, axes.T):
        plot_moments(res, raw=True, mini=True, axes=axes)

    fig.savefig("propagation_icnn.svg")
    fig.show()

    net = torch.nn.Sequential(
        nn.Linear(200, 400),
        nn.Sequential(nn.ReLU(), ConvexLinear(400, 300, positivity=LazyClippedPositivity())),
        nn.Sequential(nn.ReLU(), ConvexLinear(300, 400, positivity=LazyClippedPositivity())),
        nn.Sequential(nn.ReLU(), ConvexLinear(400, 100, positivity=LazyClippedPositivity())),
    )

    # principled init
    with torch.no_grad():
        net[0].weight -= net[0].weight.mean(1, keepdims=True)
        net[0].weight /= (200 * net[0].weight.var(1, keepdims=True)) ** .5
    for layer in net[1:]:
        layer[1].reset_parameters()

    with torch.no_grad():
        s, = ours_results = [x]
        for layer in net:
            s = layer(s)
            ours_results.append(s)

    size = (3, len(ours_results) - 1)
    fig, axes = plt.subplots(*size, figsize=(size[1] * 6.4, size[0] * 4.8))
    axes[0, 0].get_shared_y_axes().join(*axes[0])
    for res, axes in zip(ours_results, axes.T):
        plot_moments(res, raw=True, mini=True, axes=axes)

    fig.savefig("propagation_icnn_init.svg")
    fig.show()

    # bonus visualisation

    net = LinearSkip(200, 100, nn.Sequential(
        LinearSkip(200, 400, nn.Sequential(
            LinearSkip(200, 300, nn.Sequential(
                nn.Linear(200, 400),
                nn.Sequential(nn.ReLU(), nn.Linear(400, 300))
            )),
            nn.Sequential(nn.ReLU(), nn.Linear(300, 400))
        )),
        nn.Sequential(nn.ReLU(), nn.Linear(400, 100))
    ))

    with torch.no_grad():
        for name, par in net.named_parameters():
            if "residual.1.1" in name and par.ndim > 1:
                print(name)
                torch.relu_(par)

    with torch.no_grad():
        s, = skip_results = [x]
        s = net.residual[0].residual[0].residual[0](s)
        skip_results.append(s)
        s = net.residual[0].residual[0].residual[1](s) + net.residual[0].residual[0].skip(x)
        skip_results.append(s)
        s = net.residual[0].residual[1](s) + net.residual[0].skip(x)
        skip_results.append(s)
        s = net.residual[1](s) + net.skip(x)
        skip_results.append(s)

    size = (3, len(skip_results) - 1)
    fig, axes = plt.subplots(*size, figsize=(size[1] * 6.4, size[0] * 4.8))
    axes[0, 0].get_shared_y_axes().join(*axes[0])
    for res, axes in zip(skip_results, axes.T):
        plot_moments(res, raw=True, mini=True, axes=axes)

    fig.savefig("propagation_icnn_skip.svg")
    fig.show()

    # size = (1, len(nn_results) - 1)
    # fig, axes = plt.subplots(*size, sharey="row",
    #                          figsize=(size[1] * 6.4, 4.8))
    # for nn_res, icnn_res, skip_res, ours_res, ax in zip(
    #         nn_results, icnn_results, skip_results, ours_results, axes.T
    # ):
    #     ax.hist(nn_res.ravel(), color="lightgray", density=True)
    #     ax.hist(icnn_res.ravel(), color=plt.cm.tab20c(2), density=True)
    #     ax.hist(skip_res.ravel(), color=plt.cm.tab20(1), density=True)
    #     ax.hist(ours_res.ravel(), color=plt.cm.tab20(0), density=True)
    # fig.show()

