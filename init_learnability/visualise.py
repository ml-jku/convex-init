from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from upsilonconf import load_config


def collect_results(path: str, filters: dict = None, tag: str = "train/avg_loss"):
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    path = Path(path)
    results = {}
    counter = 0
    for sub_path in path.iterdir():
        hparams = load_config(sub_path / "config.yaml")
        if filters is None or all(str(v) == str(hparams.get(k)) for k, v in filters.items()):
            counter += 1
            key = (
                hparams.data.name.upper(),
                hparams.model.num_hidden,
                hparams.model.positivity or "",
                hparams.model.better_init,
                hparams.model.get("rand_bias", False),
                hparams.model.get("skip", False),
                hparams.model.get("bias_init_only", False),
                hparams.model.get("corr", 0.5)
            )
            event_file = next(sub_path.glob("events.out.tfevents.*"))
            results.setdefault(key, []).append([
                s.value
                for s in EventAccumulator(str(event_file)).Reload().Scalars(tag)
            ])

    print(f"read {counter:d} event files")
    assert counter // 10 == len(results), {k: len(v) for k, v in results.items()}
    return {k: np.array(results[k]) for k in sorted(results.keys())}


def visualise_results(data: dict[tuple[str, int, str, bool, bool, bool, bool, float], np.ndarray],
                      scale: str = "log", zoom: str = "in"):
    dataset_options = tuple(sorted({k[0] for k in data.keys()}, key=len))
    depth_options = tuple(sorted({k[1] for k in data.keys()}))
    nrows, ncols = len(depth_options), len(dataset_options)
    fig = plt.figure(figsize=(ncols * 5.6, nrows * 4.2))
    axes = fig.subplots(nrows, ncols, squeeze=False)
    for ax in axes.flat:
        if scale is not None:
            ax.set_yscale(scale)
        if zoom is not None:
            axins = ax.inset_axes([.45, .52, .5, .4])
            axins.patch.set_alpha(0.9)
            axins.get_xaxis().set_visible(False)
            axins.set_yscale(ax.get_yscale())

    label_colors = {
        ("", True, False, False, False): ("non-convex", "gray"),
        ("icnn", True, False, False, False): ("ICNN + init", "#0084bb"),
        ("icnn", False, False, True, False): ("ICNN + skip", plt.cm.tab10(1)),
        ("icnn", False, False, False, False): ("ICNN", plt.cm.tab10(2)),
        ("icnn", True, True, False, False): ("ICNN + bias-init", plt.cm.tab10(3)),
        ("icnn", True, False, False, True): ("ICNN + init (bias-only)", plt.cm.tab10(4)),
        ("exp", True, False, False, False): ("exp-ICNN + init", "#0084bb"),
        ("exp", False, False, True, False): ("exp-ICNN + skip", plt.cm.tab10(1)),
        ("exp", False, False, False, False): ("exp-ICNN", plt.cm.tab10(2)),
        ("clip", True, False, False, False): ("clip-ICNN + init", "#0084bb"),
        ("clip", False, False, True, False): ("clip-ICNN + skip", plt.cm.tab10(1)),
        ("clip", False, False, False, False): ("clip-ICNN", plt.cm.tab10(2)),
    }

    all_positivities = set()
    for k, v in data.items():
        dataset_name, num_hidden, positivity, best_init, rand_bias, skip, bias_init_only, corr = k
        all_positivities.add(positivity)
        if best_init and skip:
            continue
        ax = axes[
            depth_options.index(num_hidden),
            dataset_options.index(dataset_name)
        ]
        lbl, col = label_colors[positivity, best_init, rand_bias, skip, bias_init_only]
        if corr != 0.5:
            col = plt.cm.tab10(5) if corr > 0.5 else plt.cm.tab10(6)
            lbl = f"{lbl} (corr: {corr:.1f})"

        take_every = 50 if v.shape[1] > 5000 else 1
        x = range(0, v.shape[1], take_every)
        v = v[:, ::take_every]
        q0, q1, q2, q3, q4 = np.quantile(v, [0., .25, .5, .75, 1.], axis=0)

        ax.fill_between(x, q1, q3, color=col, alpha=.5, zorder=1)
        ax.plot(x, q2, color=col, label=lbl, zorder=3)
        # ax.plot(x, q0, "--", q4, "--", color=col, zorder=2)

        if zoom is not None:
            # duplicate plot
            axins = ax.child_axes[0]
            axins.fill_between(x, q1, q3, color=col, alpha=.5, zorder=1)
            axins.plot(x, q2, color=col, label=lbl, zorder=3)
            # axins.plot(x, q0, "--", q4, "--", color=col, zorder=2)

            zoom_in_ax = axins if zoom == "in" else ax
            if positivity == "":
                zoom_in_ax.set_ylim(None, 1.05 * q4.max())
            elif zoom_in_ax.get_ylim()[0] > q0.min():
                zoom_in_ax.set_ylim(0.95 * q0.min(), None)
            # ax.indicate_inset_zoom(axins)
            # if positivity == "":
            # #     axins.set_xlim(0, v.shape[1] - 1)
            # #     axins.set_ylim(.75 * q0.max(), None)
            # # elif axins.get_ylim()[1] < q4.max():
            # #     axins.set_ylim(None, q4.max())
            #     axins.set_ylim(None, .95 * q4.max())
            # elif axins.get_ylim()[0] > q0.min():
            #     axins.set_ylim(q0.min(), None)

    vertical = "lower" if zoom == "out" else "upper"
    horizontal = "right" if zoom is None else "left"
    axes[0, 0].legend(loc=f"{vertical} {horizontal}")
    axes[0, 0].set_ylabel("training loss")
    for ax, col_title in zip(axes[0, :], dataset_options):
        ax.set_title(col_title, fontdict={"fontsize": "x-large"})
    if axes.shape[0] > 1:
        for ax, depth in zip(axes[:, 0], depth_options):
            ax.set_ylabel(f"{depth} hidden layers")
    # for ax, depth in zip(axes[:, 0], depth_options):
    #     ax.text(-.1, .5, f"{depth + 1}-layer net", transform=ax.transAxes,
    #             ha="right", va="center", rotation="vertical")
    if zoom is not None and all_positivities == {"", "icnn"}:
        for ax in axes.flat:
            bottom, top = ax.get_ylim()
            ax.set_ylim(None, top ** .5)

    fig.tight_layout()
    return fig


if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("dir", type=Path, help="directory with results")
    parser.add_argument("--convexity", type=str, default=None,
                        choices=["icnn", "exp", "clip"])
    parser.add_argument("--tb-tag", type=str, default="train/batch_loss",
                        help="tensorboard tag to visualise")
    args = parser.parse_args()

    filters = {}
    if args.convexity is not None:
        filters["model.positivity"] = args.convexity

    plot_scale = "linear" if "acc" in args.tb_tag else "log"
    ref_data = collect_results(args.dir, tag=args.tb_tag, filters={"model.positivity": "None"})
    data = ref_data | collect_results(args.dir, tag=args.tb_tag, filters=filters)
    visualise_results(data, scale=plot_scale).show()
