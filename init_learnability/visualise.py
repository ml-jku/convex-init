from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from upsilonconf import load_config


def collect_results(path: str, filters: dict = None, tag: str = "train/avg_loss"):
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    path = Path(path)
    results = {}
    for sub_path in path.iterdir():
        hparams = load_config(sub_path / "config.yaml")
        if filters is None or all(str(v) in str(hparams[k]) for k, v in filters.items()):
            key = (
                hparams.data.name.upper(),
                hparams.model.num_hidden,
                hparams.model.positivity or "",
                hparams.model.better_init,
                hparams.model.get("skip", False),
            )
            event_file = next(sub_path.glob("events.out.tfevents.*"))
            results.setdefault(key, []).append([
                s.value
                for s in EventAccumulator(str(event_file)).Reload().Scalars(tag)
            ])

    return {k: np.array(results[k]) for k in sorted(results.keys())}


def visualise_results(data: dict[tuple[str, int, str, bool, bool], np.ndarray],
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
        ("", True, False): ("non-convex", "lightgray"),
        ("icnn", True, False): ("ICNN + init", plt.cm.tab20c(0)),
        ("icnn", False, True): ("ICNN + skip", plt.cm.tab20c(1)),
        ("icnn", False, False): ("ICNN", plt.cm.tab20c(2)),
        ("exp", True, False): ("exp-ICNN + init", plt.cm.tab20c(4)),
        ("exp", False, True): ("exp-ICNN + skip", plt.cm.tab20c(5)),
        ("exp", False, False): ("exp-ICNN", plt.cm.tab20c(6)),
        ("clip", True, False): ("clip-ICNN + init", plt.cm.tab20c(8)),
        ("clip", False, True): ("clip-ICNN + skip", plt.cm.tab20c(9)),
        ("clip", False, False): ("clip-ICNN", plt.cm.tab20c(10)),
    }

    for k, v in data.items():
        dataset_name, num_hidden, positivity, best_init, skip = k
        if positivity == "clip" or best_init and skip:
            continue
        ax = axes[
            depth_options.index(num_hidden),
            dataset_options.index(dataset_name)
        ]
        lbl, col = label_colors[positivity, best_init, skip]

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

    axes[0, 0].legend(loc="lower left")
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
