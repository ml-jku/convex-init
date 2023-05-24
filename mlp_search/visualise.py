from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt


def retrieve_five_number_summary(path: Path, pattern: str, tb_tag: str = "train/avg_loss"):
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    results = []
    for p in path.glob(pattern):
        if p.is_dir():
            event_path = next(p.glob(f"events.out.tfevents.*"))
            ea = EventAccumulator(str(event_path))
            events = ea.Reload().Scalars(tb_tag)
            results.append([e.value for e in events])

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


def generate_mnist_plot():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.4, 4.2))
    result_path = Path("results", "mlp_search")

    # non-convex:      mnist071
    # exp-icnn:        convex1_mnist023
    # exp-icnn + init: convex_mnist066
    # icnn:            icnn1_mnist0?? (icnn_mnist047)
    # icnn + init:     icnn2_mnist052

    stats = retrieve_five_number_summary(result_path, "mnist???.?_*", tb_tag="valid/loss")
    plot_five_number_summary(stats, "non-convex", color="gray", ax=ax1)
    # stats = retrieve_five_number_summary(result_path, "convex1_mnist???.?_*", tb_tag="valid/loss")
    # plot_five_number_summary(stats, "exp-ICNN", color=plt.cm.tab10(2), ax=ax1)
    # stats = retrieve_five_number_summary(result_path, "convex_mnist???.?_*", tb_tag="valid/loss")
    # plot_five_number_summary(stats, "exp-ICNN + init", color="#0084bb", ax=ax1)
    stats = retrieve_five_number_summary(result_path, "icnn1_mnist???.?_*", tb_tag="valid/loss")
    plot_five_number_summary(stats, "ICNN", color=plt.cm.tab10(2), ax=ax1)
    stats = retrieve_five_number_summary(result_path, "skip_mnist???.?_*", tb_tag="valid/loss")
    plot_five_number_summary(stats, "ICNN + skip", color=plt.cm.tab10(1), ax=ax1)
    stats = retrieve_five_number_summary(result_path, "icnn2_mnist???.?_*", tb_tag="valid/loss")
    plot_five_number_summary(stats, "ICNN + init", color="#0084bb", ax=ax1)
    ax1.set_ylim(None, 1.)
    ax1.set_ylabel("average loss")
    ax1.legend()

    stats = retrieve_five_number_summary(result_path, "mnist???.?_*", tb_tag="valid/acc")
    plot_five_number_summary(stats, "non-convex", color="gray", ax=ax2)
    # stats = retrieve_five_number_summary(result_path, "convex1_mnist???.?_*", tb_tag="valid/acc")
    # plot_five_number_summary(stats, "exp-ICNN", color=plt.cm.tab10(2), ax=ax2)
    # stats = retrieve_five_number_summary(result_path, "convex_mnist???.?_*", tb_tag="valid/acc")
    # plot_five_number_summary(stats, "exp-ICNN + init", color="#0084bb", ax=ax2)
    stats = retrieve_five_number_summary(result_path, "icnn1_mnist???.?_*", tb_tag="valid/acc")
    plot_five_number_summary(stats, "ICNN", color=plt.cm.tab10(2), ax=ax2)
    stats = retrieve_five_number_summary(result_path, "skip_mnist???.?_*", tb_tag="valid/acc")
    plot_five_number_summary(stats, "ICNN + skip", color=plt.cm.tab10(1), ax=ax2)
    stats = retrieve_five_number_summary(result_path, "icnn2_mnist???.?_*", tb_tag="valid/acc")
    plot_five_number_summary(stats, "ICNN + init", color="#0084bb", ax=ax2)
    ax2.yaxis.tick_right()
    ax2.yaxis.set_label_position("right")
    ax2.set_yscale("linear")
    ax2.set_ylim(0.9, 1.0)
    ax2.set_ylabel("accuracy")

    # ax1.set_title("baseline hyper-parameters")
    # base = retrieve_five_number_summary(result_path, "mnist071?_*")
    # plot_five_number_summary(base, "baseline", ax=ax1)
    # icnn = retrieve_five_number_summary(result_path, "mnist071icnn?_*")
    # plot_five_number_summary(icnn, "ICNN", ax=ax1)
    # conv = retrieve_five_number_summary(result_path, "mnist071convex?_*")
    # plot_five_number_summary(conv, "convex", ax=ax1)
    # ax1.legend()

    # ax2.set_title("ICNN hyper-parameters")
    # base = retrieve_five_number_summary(result_path, "icnn_mnist002base?_*")
    # plot_five_number_summary(base, "baseline", ax=ax2)
    # icnn = retrieve_five_number_summary(result_path, "icnn_mnist002?_*")
    # plot_five_number_summary(icnn, "ICNN", ax=ax2)
    # conv = retrieve_five_number_summary(result_path, "icnn_mnist002convex?_*")
    # plot_five_number_summary(conv, "convex", ax=ax2)

    # ax3.set_title("convex hyper-parameters")
    # base = retrieve_five_number_summary(result_path, "convex_mnist066base?_*")
    # plot_five_number_summary(base, "baseline", ax=ax3)
    # icnn = retrieve_five_number_summary(result_path, "convex_mnist066icnn?_*")
    # plot_five_number_summary(icnn, "ICNN", ax=ax3)
    # conv = retrieve_five_number_summary(result_path, "convex_mnist066?_*")
    # plot_five_number_summary(conv, "convex", ax=ax3)

    fig.tight_layout()
    return fig


def generate_cifar_plot():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.4, 4.2))
    result_path = Path("results", "mlp_search")

    # non-convex:      cifar107
    # exp-icnn:        convex1_cifar023
    # exp-icnn + init: convex_cifar044
    # icnn:            icnn1_cifar023 (icnn_cifar022)
    # icnn + init:     icnn2_cifar022

    stats = retrieve_five_number_summary(result_path, "cifar???.?_*", tb_tag="valid/loss")
    plot_five_number_summary(stats, "non-convex", color="gray", ax=ax1)
    # stats = retrieve_five_number_summary(result_path, "convex1_cifar???.?_*", tb_tag="valid/loss")
    # plot_five_number_summary(stats, "exp-ICNN", color=plt.cm.tab10(2), ax=ax1)
    # stats = retrieve_five_number_summary(result_path, "convex_cifar???.?_*", tb_tag="valid/loss")
    # plot_five_number_summary(stats, "exp-ICNN + init", color="#0084bb", ax=ax1)
    stats = retrieve_five_number_summary(result_path, "icnn1_cifar???.?_*", tb_tag="valid/loss")
    plot_five_number_summary(stats, "ICNN", color=plt.cm.tab10(2), ax=ax1)
    stats = retrieve_five_number_summary(result_path, "skip_cifar???.?_*", tb_tag="valid/loss")
    plot_five_number_summary(stats, "ICNN + skip", color=plt.cm.tab10(1), ax=ax1)
    stats = retrieve_five_number_summary(result_path, "icnn2_cifar???.?_*", tb_tag="valid/loss")
    plot_five_number_summary(stats, "ICNN + init", color="#0084bb", ax=ax1)
    ax1.set_ylim(None, 6.)
    ax1.set_ylabel("average loss")
    ax1.legend()

    stats = retrieve_five_number_summary(result_path, "cifar???.?_*", tb_tag="valid/acc")
    plot_five_number_summary(stats, "non-convex", color="gray", ax=ax2)
    # stats = retrieve_five_number_summary(result_path, "convex1_cifar???.?_*", tb_tag="valid/acc")
    # plot_five_number_summary(stats, "exp-ICNN", color=plt.cm.tab10(2), ax=ax2)
    # stats = retrieve_five_number_summary(result_path, "convex_cifar???.?_*", tb_tag="valid/acc")
    # plot_five_number_summary(stats, "exp-ICNN + init", color="#0084bb", ax=ax2)
    stats = retrieve_five_number_summary(result_path, "icnn1_cifar???.?_*", tb_tag="valid/acc")
    plot_five_number_summary(stats, "ICNN", color=plt.cm.tab10(2), ax=ax2)
    stats = retrieve_five_number_summary(result_path, "skip_cifar???.?_*", tb_tag="valid/acc")
    plot_five_number_summary(stats, "ICNN", color=plt.cm.tab10(1), ax=ax2)
    stats = retrieve_five_number_summary(result_path, "icnn2_cifar???.?_*", tb_tag="valid/acc")
    plot_five_number_summary(stats, "ICNN + init", color="#0084bb", ax=ax2)
    ax2.yaxis.tick_right()
    ax2.yaxis.set_label_position("right")
    ax2.set_yscale("linear")
    ax2.set_ylim(0.3, 0.6)
    ax2.set_ylabel("accuracy")

    # ax1.set_title("baseline hyper-parameters")
    # base = retrieve_five_number_summary(result_path, "cifar_base_rep?_*")
    # plot_five_number_summary(base, "baseline", ax=ax1)
    # icnn = retrieve_five_number_summary(result_path, "cifar_icnn_rep?_*")
    # plot_five_number_summary(icnn, "ICNN", ax=ax1)
    # conv = retrieve_five_number_summary(result_path, "cifar_conv_rep?_*")
    # plot_five_number_summary(conv, "convex", ax=ax1)
    # ax1.legend()

    fig.tight_layout()
    return fig
