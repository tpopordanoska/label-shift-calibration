import random
from os.path import join as pjoin

import numpy as np
import torch
import yaml


def format_mean_std(value, std=None):
    if not std:
        return f"${(100 * value):.2f}$"
    return f"${(100 * value):.2f}_{{\\pm {(100 * std):.2f}}}$"


def set_random_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def load_experiment_config(dir_path):
    with open(pjoin(dir_path, "config.yaml"), "r") as file:
        config = yaml.safe_load(file)
    return config
