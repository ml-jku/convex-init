from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt


def retrieve_five_number_summary(path: Path, pattern: str, tb_tag: str = "train/avg_loss"):
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    results = []
    for p in path.glob(pattern):
        if p.is_dir():
            config, _ = p.name.rsplit(".", maxsplit=1)
            search_path = next(p.parent.glob(config + "_*/events.out.tfevents.*"))
            valid_ea = EventAccumulator(str(search_path))
            acc_events = valid_ea.Reload().Scalars("valid/acc")
            early_stop = max(acc_events, key=lambda e: e.value).step

            event_path = next(p.glob(f"events.out.tfevents.*"))
            ea = EventAccumulator(str(event_path))
            events = ea.Reload().Scalars(tb_tag)
            result = [e.value for e in events[:early_stop]]
            padding = [events[early_stop].value] * (len(events) - early_stop)
            assert len(result) + len(padding) == len(events)
            results.append(result + padding)

    return np.quantile(results, q=[0., .25, .5, .75, 1.], axis=0)


def plot_five_number_summary(summary, label=None, color=None, ax=None):
    if len(summary) != 5:
        raise ValueError("not a five-number summary")
    if ax is None:
        ax = plt.gca()
    if color is None:
        color = next(ax._get_lines.prop_cycler)["color"]

    # lines = ax.semilogy(summary[[0, -1]].T, "--", color=color)
    # color = lines[0].get_color()
    ax.fill_between(range(summary.shape[1]), summary[1], summary[-2],
                    alpha=0.5, color=color)
    ax.semilogy(summary[2], color=color, label=label)
    return ax


def generate_mnist_plot(identifiers: list[str], id_plot_kwargs: dict[str, tuple]):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.4, 4.2))
    result_path = Path("results", "mlp_search")

    # non-convex:      nn_mnist071
    # exp-icnn:        exp1_mnist023
    # exp-icnn + init: exp2_mnist066
    # icnn:            icnn_mnist047
    # icnn + init:     ours_mnist052

    for key in identifiers:
        label, col = id_plot_kwargs[key]
        stats = retrieve_five_number_summary(result_path, f"{key}_mnist???.?_*", tb_tag="valid/loss")
        plot_five_number_summary(stats, label, color=col, ax=ax1)
        stats = retrieve_five_number_summary(result_path, f"{key}_mnist???.?_*", tb_tag="valid/acc")
        plot_five_number_summary(stats, label, color=col, ax=ax2)

    ax1.set_ylim(None, 1.)
    ax1.set_ylabel("average loss")
    ax1.legend()

    ax2.yaxis.tick_right()
    ax2.yaxis.set_label_position("right")
    ax2.set_yscale("linear")
    ax2.set_ylim(0.9, 1.0)
    ax2.set_ylabel("accuracy")

    fig.tight_layout()
    return fig


def generate_cifar_plot(identifiers: list[str], id_plot_kwargs: dict[str, tuple]):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.4, 4.2))
    result_path = Path("results", "mlp_search")

    # non-convex:      nn_cifar107
    # exp-icnn:        exp1_cifar023
    # exp-icnn + init: exp2_cifar044
    # icnn:            icnn_cifar022
    # icnn + init:     ours_cifar022

    for key in identifiers:
        label, col = id_plot_kwargs[key]
        stats = retrieve_five_number_summary(result_path, f"{key}_cifar???.?_*", tb_tag="valid/loss")
        plot_five_number_summary(stats, label, color=col, ax=ax1)
        stats = retrieve_five_number_summary(result_path, f"{key}_cifar???.?_*", tb_tag="valid/acc")
        plot_five_number_summary(stats, label, color=col, ax=ax2)

    ax1.set_ylim(None, 6.)
    ax1.set_ylabel("average loss")
    ax1.legend()

    ax2.yaxis.tick_right()
    ax2.yaxis.set_label_position("right")
    ax2.set_yscale("linear")
    ax2.set_ylim(0.3, 0.6)
    ax2.set_ylabel("accuracy")

    fig.tight_layout()
    return fig


if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("identifiers", metavar="ID", type=str, nargs="+",
                        description="model identifiers to plot the results for")
    args = parser.parse_args()

    id_colours = {
        "nn": ("non-convex", "gray"),
        "icnn": ("ICNN", plt.cm.tab10(2)),
        "skip": ("ICNN + skip", plt.cm.tab10(1)),
        "ours": ("ICNN + init", "#0084bb"),
        "exp1": ("exp-ICNN", plt.cm.tab10(2)),
        "exp2": ("exp-ICNN + init", "#0084bb"),
    }

    generate_mnist_plot(args.identifiers, id_colours)
    generate_cifar_plot(args.identifiers, id_colours)
