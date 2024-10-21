import warnings

import numpy as np
import torch
from abstention.calibration import TempScaling
from abstention.label_shift import (
    BBSEImbalanceAdapter,
    EMImbalanceAdapter,
    RLLSImbalanceAdapter,
)
from torch import nn
from tqdm.auto import tqdm

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
    # valid_labels_numerical = np.argmax(valid_labels, axis=1)
    # shifted_test_labels_numerical = np.argmax(shifted_test_labels, axis=1)
    # true_weight = compute_true_w(
    #     valid_labels_numerical,
    #     shifted_test_labels_numerical,
    #     n_class=valid_preds.shape[1],
    # )
    # print("true empirical weight: ", true_weight)
    # print(f"{method} error: {calculate_angle(true_weight, imbalance_adapter_func.multipliers)}")

    return {
        "weights": imbalance_adapter_func.multipliers,
        "adapted_test_pred": adapted_shifted_test_preds,
    }


def get_bin_specific_weights(
    f_source,
    f_target,
    y_source,
    y_target,
    method,
    adaptive_bins=True,
    n_bins=15,
    classwise=False,
):
    def get_bin_weights(
        bin_lowers, bin_uppers, f_source, f_target, y_source_ohe, y_target_ohe
    ):
        weights = []
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin_source = f_source.gt(bin_lower.item()) * f_source.le(
                bin_upper.item()
            )
            in_bin_target = f_target.gt(bin_lower.item()) * f_target.le(
                bin_upper.item()
            )
            conf_source_in_bin = f_source[in_bin_source]
            conf_source_in_bin = torch.stack(
                (1 - conf_source_in_bin, conf_source_in_bin), dim=1
            )
            y_source_ohe_in_bin = y_source_ohe[in_bin_source]
            conf_target_in_bin = f_target[in_bin_target]
            conf_target_in_bin = torch.stack(
                (1 - conf_target_in_bin, conf_target_in_bin), dim=1
            )
            y_target_ohe_in_bin = y_target_ohe[in_bin_target]
            l = get_weights(
                conf_source_in_bin.numpy(),
                y_source_ohe_in_bin.numpy(),
                conf_target_in_bin.numpy(),
                y_target_ohe_in_bin.numpy(),
                method,
            )
            weights.append(l[1])
        return torch.tensor(weights)

    if classwise:
        num_classes = f_source.shape[1]
        classwise_weights = []
        for i in range(num_classes):
            class_confidences_source = f_source[:, i]
            class_confidences_target = f_target[:, i]
            class_labels_source = y_source.eq(i)
            class_labels_source_ohe = nn.functional.one_hot(
                class_labels_source.to(int), num_classes=2
            ).float()
            class_labels_target = y_target.eq(i)
            class_labels_target_ohe = nn.functional.one_hot(
                class_labels_target.to(int), num_classes=2
            ).float()
            bin_lowers, bin_uppers = set_bins(
                class_confidences_target, adaptive_bins=adaptive_bins, n_bins=n_bins
            )
            weights = get_bin_weights(
                bin_lowers,
                bin_uppers,
                class_confidences_source,
                class_confidences_target,
                class_labels_source_ohe,
                class_labels_target_ohe,
            )
            classwise_weights.append(weights)
    else:
        bin_lowers, bin_uppers = set_bins(
            f_target[:, 1], adaptive_bins=adaptive_bins, n_bins=n_bins
        )
        y_source_ohe = nn.functional.one_hot(y_source.to(int), num_classes=2).float()
        y_target_ohe = nn.functional.one_hot(y_target.to(int), num_classes=2).float()
        weights = get_bin_weights(
            bin_lowers,
            bin_uppers,
            f_source[:, 1],
            f_target[:, 1],
            y_source_ohe,
            y_target_ohe,
        )

    return torch.stack(classwise_weights).transpose(0, 1) if classwise else weights


def get_gt_bin_specific_weights(
    conf_source,
    conf_target,
    labels_source,
    labels_target,
    adaptive_bins=True,
    n_bins=15,
    classwise=False,
):
    def get_bin_weights(
        bin_lowers, bin_uppers, f_source, f_target, labels_source, labels_target
    ):
        weights = []
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin_source = f_source.gt(bin_lower.item()) * f_source.le(
                bin_upper.item()
            )
            in_bin_target = f_target.gt(bin_lower.item()) * f_target.le(
                bin_upper.item()
            )
            p_labels_source = labels_source[in_bin_source].sum() / len(labels_source)
            p_labels_target = labels_target[in_bin_target].sum() / len(labels_target)
            weight = p_labels_target / p_labels_source if p_labels_source > 0 else 0.0
            weights.append(weight)
        return torch.tensor(weights)

    if classwise:
        num_classes = conf_source.shape[1]
        classwise_weights = []
        for i in tqdm(range(num_classes)):
            class_confidences_source = conf_source[:, i]
            class_confidences_target = conf_target[:, i]
            class_labels_source = labels_source.eq(i)
            class_labels_target = labels_target.eq(i)
            bin_lowers, bin_uppers = set_bins(
                class_confidences_target, adaptive_bins=adaptive_bins, n_bins=n_bins
            )
            weights = get_bin_weights(
                bin_lowers,
                bin_uppers,
                class_confidences_source,
                class_confidences_target,
                class_labels_source,
                class_labels_target,
            )
            classwise_weights.append(weights)
    else:
        bin_lowers, bin_uppers = set_bins(
            conf_target[:, 1], adaptive_bins=adaptive_bins, n_bins=n_bins
        )
        weights = get_bin_weights(
            bin_lowers,
            bin_uppers,
            conf_source[:, 1],
            conf_target[:, 1],
            labels_source,
            labels_target,
        )

    # classwise_weights shape [n_bins, n_classes]
    return torch.stack(classwise_weights).transpose(0, 1) if classwise else weights
