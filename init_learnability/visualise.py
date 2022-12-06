from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from upsilonconf import load_config


def collect_results(path: str, filters: dict = None, tag: str = "train/avg_loss"):
    from tensorboard.backend.event_processing import event_accumulator

    path = Path(path)
    results = {}
    for sub_path in path.iterdir():
        hparams = load_config(sub_path / "config.yaml")
        if filters is None or all(str(v) in str(hparams[k]) for k, v in filters.items()):
            event_file = next(sub_path.glob("events.out.tfevents.*"))
            ea = event_accumulator.EventAccumulator(str(event_file))
            scalars = ea.Reload().Scalars(tag)

            key = (
                hparams.data.name,
                hparams.model.num_hidden,
                hparams.model.positivity or "",
                hparams.model.better_init,
            )
            res = results.setdefault(key, [])
            res.append([s.value for s in scalars])

    return {k: np.array(results[k]) for k in sorted(results.keys())}


def visualise_results(data: dict[tuple[str, int, str, bool], np.ndarray],
                      scale: str = "log"):
    dataset_options = tuple(sorted({k[0] for k in data.keys()}, key=len))
    depth_options = tuple(sorted({k[1] for k in data.keys()}))
    nrows, ncols = len(depth_options), len(dataset_options)
    fig = plt.figure(figsize=(ncols * 6.4, nrows * 4.8))
    axes = fig.subplots(nrows, ncols, squeeze=False)
    for ax in axes.flat:
        axins = ax.inset_axes([.5, .3, .45, .35])
        axins.get_xaxis().set_visible(False)
        if scale is not None:
            ax.set_yscale(scale)
            axins.set_yscale(scale)

    label_colors = {
        ("", True): ("non-convex", "gray"),
        ("clip", False): ("clip raw", plt.cm.Paired(0)),
        ("clip", True): ("clip init", "white"),
        ("exp", False): ("exp raw", plt.cm.Paired(2)),
        ("exp", True): ("exp init", plt.cm.Paired(3)),
    }

    for k, v in data.items():
        dataset_name, num_hidden, positivity, best_init = k
        if positivity == "clip" and best_init is True:
            continue
        ax = axes[
            depth_options.index(num_hidden),
            dataset_options.index(dataset_name)
        ]
        lbl, col = label_colors[positivity, best_init]

        x = range(v.shape[1])
        q0, q1, q2, q3, q4 = np.quantile(v, [0., .25, .5, .75, 1.], axis=0)

        ax.fill_between(x, q1, q3, color=col, alpha=.5, zorder=1)
        ax.plot(q2, color=col, label=lbl, zorder=3)
        ax.plot(x, q0, "--", q4, "--", color=col, zorder=2)

        # zoomed
        axins = ax.child_axes[0]
        axins.fill_between(x, q1, q3, color=col, alpha=.5, zorder=1)
        axins.plot(q2, color=col, label=lbl, zorder=3)
        axins.plot(x, q0, "--", q4, "--", color=col, zorder=2)
        if positivity == "":
        #     axins.set_xlim(0, v.shape[1] - 1)
        #     axins.set_ylim(.75 * q0.max(), None)
        # elif axins.get_ylim()[1] < q4.max():
        #     axins.set_ylim(None, q4.max())
            axins.set_ylim(None, .95 * q4.max())
        elif axins.get_ylim()[0] > q0.min():
            axins.set_ylim(q0.min(), None)

    for ax in axes.flat:
        axins = ax.child_axes[0]
        ax.indicate_inset_zoom(axins)

    axes[0, 0].legend()
    for ax, col_title in zip(axes[0, :], dataset_options):
        ax.set_title(col_title)
    for ax, depth in zip(axes[:, 0], depth_options):
        ax.text(-.1, .5, f"{depth + 1}-layer net", transform=ax.transAxes,
                ha="right", va="center", rotation="vertical")
    return fig


if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("dir", type=Path, help="directory with results")
    parser.add_argument("--tb-tag", type=str, default="train/avg_loss",
                        help="tensorboard tag to visualise")
    args = parser.parse_args()

    plot_scale = "linear" if "acc" in args.tb_tag else "log"
    data = collect_results(args.dir, tag=args.tb_tag)
    visualise_results(data, scale=plot_scale).show()
