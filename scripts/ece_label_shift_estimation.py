import argparse
import json
from os.path import exists as pexists
from os.path import isdir
from os.path import join as pjoin

import numpy as np
import torch

from lascal import BootstrapMeanVarEstimator, Ece, EceLabelShift, get_importance_weights
from lascal.importance_weights import compute_true_w
from lascal.utils import SoftmaxClipper, initialize_overwatch
from lascal.utils.common import format_mean_std, set_random_seeds

# Initialize Overwatch =>> Wraps `logging.Logger`
overwatch = initialize_overwatch(__name__)


def get_acc_and_ci(y_true, y_pred, z=1.96):
    n = len(y_true)
    acc = ((y_true == y_pred).sum() / n).item()
    ci = z * np.sqrt((acc * (1 - acc)) / n)

    return acc, ci


def convert_labels_to_one_hot(logits, labels):
    num_classes = logits.shape[1]
    return np.eye(num_classes)[labels.numpy()].astype(float)


def load_prepare_predictions_labels(dir_path):
    source_agg = json.load(open(pjoin(dir_path, "source_agg.json")))
    target_agg = json.load(open(pjoin(dir_path, "target_agg.json")))

    source_logits = torch.tensor(source_agg["y_logits"])
    source_labels = torch.tensor(source_agg["y_true"])
    target_logits = torch.tensor(target_agg["y_logits"])
    target_labels = torch.tensor(target_agg["y_true"])

    return source_logits, source_labels, target_logits, target_labels


def print_classwise_table(args):
    set_random_seeds(0)
    datasets = ["cifar10", "cifar100"]
    cifar_models = ["resnet20", "resnet32", "resnet56", "resnet110"]
    models_map = {
        "resnet20": "ResNet-20",
        "resnet32": "ResNet-32",
        "resnet56": "ResNet-56",
        "resnet110": "ResNet-110",
    }
    weights_methods = ["rlls-hard", "elsa", "em-bcts", "bbse-hard"]
    # Prepare Std estimators
    ece_estimator = BootstrapMeanVarEstimator(
        estimator=Ece(adaptive_bins=True, n_bins=args.n_bins, p=args.p, classwise=True),
        reported=args.reported,
        num_bootstrap_samples=args.num_bootstrap_samples,
    )
    ece_label_shift_estimator = BootstrapMeanVarEstimator(
        estimator=EceLabelShift(
            adaptive_bins=True, n_bins=args.n_bins, p=args.p, classwise=True
        ),
        reported=args.reported,
        num_bootstrap_samples=args.num_bootstrap_samples,
    )

    softmax_clipper = SoftmaxClipper()

    for dataset in datasets:
        overwatch.info(f"Dataset: {dataset}")

        for model in cifar_models:
            string = f"{models_map[model]}"
            dir_path = pjoin(
                args.experiments_path,
                f"{dataset}_long_tail_{model}_sgd_if_{args.imbalance_factor}",
            )
            if not pexists(dir_path) or not isdir(dir_path):
                overwatch.warning(f"Path {dir_path} does not exist!")
                continue

            source_logits, source_labels, target_logits, target_labels = (
                load_prepare_predictions_labels(dir_path)
            )
            ### Accuracy source & target ###
            acc_source, acc_source_std_dev = get_acc_and_ci(
                source_labels, source_logits.argmax(-1)
            )
            acc_target, acc_target_std_dev = get_acc_and_ci(
                target_labels, target_logits.argmax(-1)
            )
            string += f" & {format_mean_std(acc_source, acc_source_std_dev)}"
            string += f" & {format_mean_std(acc_target, acc_target_std_dev)}"

            ### CE source & target ###
            ece_source, variance_source = ece_estimator(
                logits=source_logits, labels=source_labels
            )
            ece_target, variance_target = ece_estimator(
                logits=target_logits, labels=target_labels
            )
            string += f" & {format_mean_std(ece_source, np.sqrt(variance_source))}"
            string += f" & {format_mean_std(ece_target, np.sqrt(variance_target))}"

            ### CE target (w*) ###
            y_source_ohe = convert_labels_to_one_hot(source_logits, source_labels)
            y_target_ohe = convert_labels_to_one_hot(target_logits, target_labels)

            true_weight = compute_true_w(
                train_labels=np.argmax(y_source_ohe, axis=1),
                test_labels=np.argmax(y_target_ohe, axis=1),
                n_class=source_logits.shape[1],
            )
            ece_label_shift_gt_weights, variance_label_shift_gt_weight = (
                ece_label_shift_estimator(
                    logits_source=source_logits,
                    labels_source=source_labels,
                    logits=target_logits,
                    weights=true_weight,
                )
            )
            string += f" & {format_mean_std(ece_label_shift_gt_weights, np.sqrt(variance_label_shift_gt_weight))}"

            ### CE target (hat(w)}) ###
            for weights_method in weights_methods:
                fx_source, fx_target = softmax_clipper(
                    val_softmax_preds=source_logits.softmax(-1).numpy(),
                    test_softmax_preds=target_logits.softmax(-1).numpy(),
                    method=weights_method,
                )
                output = get_importance_weights(
                    fx_source,
                    y_source_ohe,
                    fx_target,
                    y_target_ohe,
                    method=weights_method,
                )
                ece_label_shift, variance_label_shift = ece_label_shift_estimator(
                    logits_source=source_logits,
                    labels_source=source_labels,
                    logits=target_logits,
                    weights=torch.tensor(output["weights"]),
                )
                string += f" & {format_mean_std(ece_label_shift, np.sqrt(variance_label_shift))}"

            string += " \\\\"
            print(string)


def main():
    parser = argparse.ArgumentParser(description="...")
    parser.add_argument("--experiments_path", type=str, required=True)
    parser.add_argument("--imbalance_factor", type=str, default=0.5)
    parser.add_argument("--p", type=int, default=2)
    parser.add_argument("--reported", type=str, default="mean")
    parser.add_argument("--num_bootstrap_samples", type=int, default=100)
    parser.add_argument("--n_bins", type=int, default=15)
    args = parser.parse_args()
    print_classwise_table(args)


if __name__ == "__main__":
    main()
