import argparse
import json
import os
from os.path import join as pjoin

import matplotlib as mpl
import numpy as np
import seaborn as sns
import tikzplotlib
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from netcal.presentation import ReliabilityDiagram

from lascal import Calibrator, Ece
from lascal.utils import initialize_overwatch

mpl.rcParams.update(mpl.rcParamsDefault)
plt.style.use("ggplot")
sns.set_palette("tab10")

# Initialize Overwatch =>> Wraps `logging.Logger`
overwatch = initialize_overwatch(__name__)

# https://github.com/nschloe/tikzplotlib/issues/567#issuecomment-1370846461
Line2D._us_dashSeq = property(lambda self: self._dash_pattern[1])
Line2D._us_dashOffset = property(lambda self: self._dash_pattern[0])


def tikzplotlib_fix_ncols(obj):
    """
    workaround for matplotlib 3.6 renamed legend's _ncol to _ncols, which breaks tikzplotlib
    """
    if hasattr(obj, "_ncols"):
        obj._ncol = obj._ncols
    for child in obj.get_children():
        tikzplotlib_fix_ncols(child)


def plot_and_save_diagram(target_agg, ece_value, title):
    target_logits = target_agg["y_logits"]
    target_labels = target_agg["y_true"]

    target_scores = F.softmax(target_logits, dim=1)

    diagram = ReliabilityDiagram(10)
    plt_d = diagram.plot(target_scores.numpy(), target_labels.numpy())
    # Remove the top subplot
    plt_d.delaxes(plt_d.axes[0])

    # Add the text with a semi-transparent white background
    plt_d.text(
        0.98,
        0.05,
        f"ECE={round(100 * ece_value, 2)}",
        transform=plt.gca().transAxes,
        fontsize=14,
        verticalalignment="bottom",
        horizontalalignment="right",
        bbox=dict(facecolor="white", alpha=0.5, edgecolor="none"),
    )

    tikzplotlib.clean_figure()
    tikzplotlib_fix_ncols(plt_d)
    tikzplotlib.save(
        os.path.join("figs", f"reliability_diagram_{title}.tex"),
        extra_groupstyle_parameters=["vertical sep=2cm"],
    )
    plt.tight_layout()
    plt_d.show()


def plot_reliability_diagram(args):
    np.random.seed(0)
    weights_methods = ["rlls-hard"]
    ece_estimator = Ece(adaptive_bins=True, n_bins=args.n_bins, p=1, classwise=False)

    dir_path = args.experiment_path
    if not os.path.exists(dir_path) or not os.path.isdir(dir_path):
        overwatch.error(f"Path {dir_path} does not exist!")

    try:
        train_agg = json.load(open(pjoin(dir_path, "train_agg.json")))
        train_agg = {k: torch.tensor(v) for k, v in train_agg.items()}
    except FileNotFoundError:
        train_agg = {}
    source_agg = json.load(open(pjoin(dir_path, "source_agg.json")))
    source_agg = {k: torch.tensor(v) for k, v in source_agg.items()}
    target_agg = json.load(open(pjoin(dir_path, "target_agg.json")))
    target_agg = {k: torch.tensor(v) for k, v in target_agg.items()}

    # Uncalibrated model
    ece_value = ece_estimator(
        logits=target_agg["y_logits"], labels=target_agg["y_true"]
    ).item()
    plot_and_save_diagram(
        target_agg, ece_value, title=f"{dir_path.split('/')[-1]}_uncal"
    )

    calibrator = Calibrator(experiment_path=dir_path)
    for method_name in ["temp_scale_source", "head_to_tail"]:
        overwatch.info("Method: ", method_name)
        calibrated_agg = calibrator.calibrate(
            method_name=method_name,
            source_agg=source_agg,
            target_agg=target_agg,
            train_agg=train_agg,
        )
        ece_value = ece_estimator(
            logits=calibrated_agg["target"]["y_logits"],
            labels=calibrated_agg["target"]["y_true"],
        ).item()
        plot_and_save_diagram(
            calibrated_agg["target"],
            ece_value,
            title=f"{dir_path.split('/')[-1]}_{method_name}",
        )

    calibrated_agg = calibrator.lascal(
        source_agg=source_agg, target_agg=target_agg, weights_method=weights_methods[0]
    )
    ece_value = ece_estimator(
        logits=calibrated_agg["target"]["y_logits"],
        labels=calibrated_agg["target"]["y_true"],
    ).item()
    plot_and_save_diagram(
        calibrated_agg["target"], ece_value, title=f"{dir_path.split('/')[-1]}_lascal"
    )


def main():
    parser = argparse.ArgumentParser(description="...")
    parser.add_argument("--experiment_path", type=str, required=True)
    parser.add_argument("--p", type=int, default=2)
    parser.add_argument("--n_bins", type=int, default=15)
    args = parser.parse_args()

    plot_reliability_diagram(args)


if __name__ == "__main__":
    main()
