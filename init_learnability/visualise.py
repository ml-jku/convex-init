from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from upsilonconf import load_config


def collect_results(path: str, filters: dict = None, tag: str = "train/avg_loss"):
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    path = Path(path)
    results = {}
    for sub_path in path.iterdir():
        hparams = load_config(next(sub_path.iterdir()) / "config.yaml")
        if filters is None or all(str(v) in str(hparams[k]) for k, v in filters.items()):
            key = (
                hparams.data.name,
                hparams.model.num_hidden,
                hparams.model.positivity or "",
                hparams.model.better_init,
            )
            results[key] = np.array([[
                s.value
                for s in EventAccumulator(str(event_file)).Reload().Scalars(tag)
            ] for event_file in sub_path.glob("*/events.out.tfevents.*")])

    return {k: results[k] for k in sorted(results.keys())}


def visualise_results(data: dict[tuple[str, int, str, bool], np.ndarray],
                      scale: str = "log"):
    dataset_options = tuple(sorted({k[0] for k in data.keys()}, key=len))
    depth_options = tuple(sorted({k[1] for k in data.keys()}))
    nrows, ncols = len(depth_options), len(dataset_options)
    fig = plt.figure(figsize=(ncols * 5.6, nrows * 4.2))
    axes = fig.subplots(nrows, ncols, squeeze=False)
    for ax in axes.flat:
        axins = ax.inset_axes([.5, .4, .45, .35])
        axins.get_xaxis().set_visible(False)
        if scale is not None:
            ax.set_yscale(scale)
            axins.set_yscale(scale)

    label_colors = {
        ("", True): ("non-convex", "lightgray"),
        ("icnn", True): ("ICNN ours", plt.cm.tab20(0)),
        ("icnn", False): ("ICNN He", plt.cm.tab20(1)),
        ("exp", True): ("exp-ICNN ours", plt.cm.tab20(2)),
        ("exp", False): ("exp-ICNN He", plt.cm.tab20(3)),
        ("clip", True): ("clip-ICNN ours", plt.cm.tab20(4)),
        ("clip", False): ("clip-ICNN He", plt.cm.tab20(5)),
    }

    for k, v in data.items():
        dataset_name, num_hidden, positivity, best_init = k
        if positivity == "clip":
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
        if positivity == "":
            ax.set_ylim(None, 1.05 * q4.max())
        elif ax.get_ylim()[0] > q0.min():
            ax.set_ylim(0.95 * q0.min(), None)

        # zoomed
        axins = ax.child_axes[0]
        axins.fill_between(x, q1, q3, color=col, alpha=.5, zorder=1)
        axins.plot(q2, color=col, label=lbl, zorder=3)
        axins.plot(x, q0, "--", q4, "--", color=col, zorder=2)
        # if positivity == "":
        # #     axins.set_xlim(0, v.shape[1] - 1)
        # #     axins.set_ylim(.75 * q0.max(), None)
        # # elif axins.get_ylim()[1] < q4.max():
        # #     axins.set_ylim(None, q4.max())
        #     axins.set_ylim(None, .95 * q4.max())
        # elif axins.get_ylim()[0] > q0.min():
        #     axins.set_ylim(q0.min(), None)

    for ax in axes.flat:
        axins = ax.child_axes[0]
        ax.indicate_inset_zoom(axins)

    axes[0, 1].legend(loc="lower left")
    for ax, col_title in zip(axes[0, :], dataset_options):
        ax.set_title(col_title)
    # for ax, depth in zip(axes[:, 0], depth_options):
    #     ax.text(-.1, .5, f"{depth + 1}-layer net", transform=ax.transAxes,
    #             ha="right", va="center", rotation="vertical")
    fig.tight_layout()
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
