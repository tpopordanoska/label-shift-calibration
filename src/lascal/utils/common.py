import random
from os.path import join as pjoin

import numpy as np
import torch
import yaml


def set_bins(confidences=None, n_bins=15, adaptive_bins=True):
    def histedges_equalN(x):
        npt = len(x)
        return np.interp(np.linspace(0, npt, n_bins + 1), np.arange(npt), np.sort(x))

    if adaptive_bins:
        _, bin_boundaries = np.histogram(
            confidences,
            histedges_equalN(confidences),
        )
    else:
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)

    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    return bin_lowers, bin_uppers


def get_mean_and_se(x):
    mean = np.mean(x)
    se = np.std(x) / np.sqrt(len(x))

    return mean, se


def get_acc(logits, labels):
    predicted_labels = torch.argmax(logits, dim=1)
    correct_predictions = (predicted_labels == labels).sum().item()
    return correct_predictions / labels.size(0)


def format_ece(value, std=None):
    if not std:
        return f"${(100 * value):.2f}$"
    return f"${(100 * value):.2f}_{{\\pm {(100 * std):.2f}}}$"


def set_random_seeds(config):
    seed = config["SEED"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def load_experiment_config(dir_path):
    with open(pjoin(dir_path, "config.yaml"), "r") as file:
        config = yaml.safe_load(file)
    return config
