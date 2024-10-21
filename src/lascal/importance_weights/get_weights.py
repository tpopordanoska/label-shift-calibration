import warnings

import numpy as np
import torch
from abstention.calibration import TempScaling
from abstention.label_shift import (
    BBSEImbalanceAdapter,
    EMImbalanceAdapter,
    RLLSImbalanceAdapter,
)

from lascal.importance_weights.elsa import EEImbalanceAdapter
from lascal.utils.common import set_bins

# cvxpy library throws warnings
warnings.filterwarnings("ignore")


def unit_vector(vector):
    """Returns the unit vector of the vector."""
    return vector / np.linalg.norm(vector)


def calculate_angle(v1, v2):
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return 1 - np.dot(v1_u, v2_u)


def compute_true_w(train_labels, test_labels, n_class=10):
    mu_y_train = np.zeros(n_class)
    train_size = len(train_labels)
    test_size = len(test_labels)
    for i in range(n_class):
        mu_y_train[i] = float(len(np.where(train_labels == i)[0])) / train_size
    mu_y_test = np.zeros(n_class)
    for i in range(n_class):
        mu_y_test[i] = float(len(np.where(test_labels == i)[0])) / test_size
    true_w = mu_y_test / mu_y_train

    return torch.tensor(true_w)


def get_weights(
    valid_preds, valid_labels, shifted_test_preds, shifted_test_labels=None, method=None
):
    if method == "em":
        imbalance_adapter = EMImbalanceAdapter()
    elif method == "em-bcts":
        imbalance_adapter = EMImbalanceAdapter(
            calibrator_factory=TempScaling(verbose=False, bias_positions="all")
        )
    elif method == "rlls-hard":
        imbalance_adapter = RLLSImbalanceAdapter()
    elif method == "rlls-soft":
        imbalance_adapter = RLLSImbalanceAdapter(soft=True)
    elif method == "bbse-hard":
        imbalance_adapter = BBSEImbalanceAdapter()
    elif method == "bbse-soft":
        imbalance_adapter = BBSEImbalanceAdapter(soft=True)
    elif method == "elsa":
        imbalance_adapter = EEImbalanceAdapter(constraint=True)
    else:
        raise NotImplementedError

    imbalance_adapter_func = imbalance_adapter(
        valid_labels=valid_labels,
        tofit_initial_posterior_probs=shifted_test_preds,
        valid_posterior_probs=valid_preds,
    )
    adapted_shifted_test_preds = imbalance_adapter_func(shifted_test_preds)

    return {
        "weights": imbalance_adapter_func.multipliers,
        "adapted_test_pred": adapted_shifted_test_preds,
    }
