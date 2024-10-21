import argparse
import json
from copy import deepcopy
from os.path import exists as pexists
from os.path import join as pjoin

import numpy as np
import torch

from lascal.calibration_error import Ece, EceLabelShift
from lascal.importance_weights import get_weights
from lascal.post_hoc_calibration import Calibrator
from lascal.utils.common import format_ece, load_experiment_config, set_random_seeds
from lascal.utils.constants import CAL_METHODS
from lascal.utils.overwatch import initialize_overwatch
from lascal.utils.softmax_clipper import SoftmaxClipper
from lascal.utils.subsampler import DatasetSubsampler
from lascal.utils.variance import BootstrapMeanVarEstimator

# Initialize Overwatch =>> Wraps `logging.Logger`
overwatch = initialize_overwatch(__name__)


class EceLabelShiftWrapper:

    def __init__(self, classwise: bool = True, adaptive_bins: bool = True, p: int = 2):
        self.classwise = classwise
        self.estimator = EceLabelShift(
            classwise=classwise, adaptive_bins=adaptive_bins, p=p
        )
        self.softmax_clipper = SoftmaxClipper()

    @torch.no_grad()
    def __call__(
        self, logits_source, labels_source, logits_target, labels_target, **kwargs
    ):
        # Transform for weight
        fx_source, fx_target = self.softmax_clipper(
            val_softmax_preds=logits_source.softmax(-1).numpy(),
            test_softmax_preds=logits_target.softmax(-1).numpy(),
            method="rlls-hard",
        )
        y_source_ohe = self.convert_labels_to_one_hot(
            logits=logits_source, labels=labels_source
        )
        y_target_ohe = self.convert_labels_to_one_hot(
            logits=logits_target, labels=labels_target
        )
        output = get_weights(
            fx_source,
            y_source_ohe,
            fx_target,
            y_target_ohe,
            method="rlls-hard",
        )
        ece = self.estimator(
            logits_source=logits_source,
            labels_source=labels_source,
            logits_target=logits_target,
            weight=output["weights"],
        )

        return ece

    @staticmethod
    def convert_labels_to_one_hot(logits, labels):
        num_classes = logits.shape[1]
        return np.eye(num_classes)[labels.numpy()].astype(float)


def report_model_result(args, path_format, dataset_name, model_name, imbalance_factor):
    string = f"{model_name} "
    dir_path = pjoin(
        args.experiments_path,
        path_format.format(
            dataset_name=dataset_name,
            model_name=model_name,
            imbalance_factor=imbalance_factor,
        ),
    )
    config = load_experiment_config(dir_path)
    # Set seeds to ensure the same source dataset
    set_random_seeds(config)

    calibrator = Calibrator(
        experiment_path=dir_path,
        verbose=args.verbose,
        covariate=args.covariate,
        criterion=args.criterion,
    )

    # Prepare ECE estimator
    estimator = Ece(classwise=not args.report_top_label, adaptive_bins=True, p=2)

    # Prepare bootstrap estimator
    bootstrap_estimator = BootstrapMeanVarEstimator(
        estimator=estimator,
        num_bootstrap_samples=args.num_bootstrap_samples,
        reported=args.reported,
    )

    # Source predictions
    org_source_agg = json.load(open(pjoin(dir_path, "source_agg.json")))
    org_source_agg = {k: torch.tensor(v) for k, v in org_source_agg.items()}
    
    # Target predictions
    target_type = "_cov_" if args.covariate else "_"
    org_target_agg = json.load(open(pjoin(dir_path, f"target{target_type}agg.json")))
    org_target_agg = {k: torch.tensor(v) for k, v in org_target_agg.items()}

    # Optional train agg
    org_train_agg = {}
    if pexists(pjoin(dir_path, "train_agg.json")):
        org_train_agg = json.load(open(pjoin(dir_path, "train_agg.json")))
        org_train_agg = {k: torch.tensor(v) for k, v in org_train_agg.items()}

    # Subsample
    dataset_subsampler = DatasetSubsampler()
    subsampled_source_agg, subsampled_target_agg = dataset_subsampler(
        dataset_name=dataset_name, source_agg=org_source_agg, target_agg=org_target_agg
    )

    results = {method_name: 0 for method_name in CAL_METHODS if method_name not in args.exclude_methods}
    for method_name in CAL_METHODS:
        if method_name in args.exclude_methods:
            continue
        
        source_agg, target_agg, train_agg = (
            deepcopy(subsampled_source_agg),
            deepcopy(subsampled_target_agg),
            deepcopy(org_train_agg),
        )

        try:
            calibrated_agg = calibrator.calibrate(
                method_name=method_name,
                source_agg=source_agg,
                target_agg=target_agg,
                train_agg=train_agg,
            )
            mean, var = bootstrap_estimator(
                logits_source=calibrated_agg["source"]["y_logits"],
                labels_source=calibrated_agg["source"]["y_true"],
                logits_target=calibrated_agg["target"]["y_logits"],
                labels_target=calibrated_agg["target"]["y_true"],
            )
            string += f"& {format_ece(mean, np.sqrt(var))} "
            results[method_name] = mean
        except np.linalg.LinAlgError:
            string += f"& -- "
        
    string += "\\\\"

    print(string)
    return results


def report_results(args, imbalance_factors, dataset_names, model_names, path_format):
    overwatch.info(f"Reporting Top-Label: {args.report_top_label}")
    overwatch.info(f"Calibrating Top-Label: {args.calibrate_top_label}")
    overwatch.info(f"Using Covariate shift data: {args.covariate}")
    for imbalance_factor in imbalance_factors:
        for dataset_name in dataset_names:
            dataset_info = f"Dataset name: {dataset_name} || Imbalance Factor: {imbalance_factor}"
            overwatch.info(dataset_info)

            # Print header
            header = "Model "
            all_results = {method_name: 0 for method_name in CAL_METHODS if method_name not in args.exclude_methods}
            for method_name in CAL_METHODS:
                if method_name in args.exclude_methods:
                    continue
                header += f"& {method_name} "

            header += "\\\\"
            print(header)

            # Print results per model
            for model_name in model_names:
                results = report_model_result(
                    args, path_format, dataset_name, model_name, imbalance_factor
                )
                for method_name in CAL_METHODS:
                    if method_name in args.exclude_methods:
                        continue
                    all_results[method_name] += results[method_name]

            # Print macro-average per method
            macro_average_latex = "\\textit{Macro average} "
            for method_name in CAL_METHODS:
                if method_name in args.exclude_methods:
                    continue
                macro_average_latex += f"& {format_ece(all_results[method_name] / len(model_names), 0)} "
            macro_average_latex += "\\\\"
            print(macro_average_latex)


def main():
    parser = argparse.ArgumentParser(description="...")
    parser.add_argument("--experiments_path", type=str, required=True)
    parser.add_argument(
        "--exclude_methods", nargs="+", default=[], help="Exclude methods..."
    )
    parser.add_argument(
        "--which_datasets",
        nargs="+",
        default=["cifar", "wilds"],
        help="For which datasets to print results",
    )
    parser.add_argument(
        "--num_bootstrap_samples",
        type=int,
        default=100,
        help="How many bootstrap samples",
    )
    parser.add_argument(
        "--reported",
        type=str,
        default="mean",
        help="Mean or sum?",
    )
    parser.add_argument(
        "--covariate",
        action="store_true",
        help="Whether to compute on covariate.",
    )
    parser.add_argument(
        "--calibrate_top_label",
        action="store_true",
        help="Whether to calibrate wtih top label ECE.",
    )
    parser.add_argument(
        "--report_top_label",
        action="store_true",
        help="Whether to report top label ECE.",
    )
    parser.add_argument(
        "--criterion",
        type=str,
        default="cross_entropy",
        help="ece or cross_entropy?",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    # Run inference
    if "cifar" in args.which_datasets:
        report_results(
            args,
            imbalance_factors=[0.1, 0.2, 0.5],
            dataset_names=["cifar10", "cifar100"],
            model_names=["resnet20", "resnet32", "resnet56", "resnet110"],
            path_format="{dataset_name}_long_tail_{model_name}_sgd_if_{imbalance_factor}",
        )
    if "wilds" in args.which_datasets:
        report_results(
            args,
            imbalance_factors=[None],
            dataset_names=["amazon"],
            model_names=["roberta", "distill_roberta_v2", "bert_v2", "distill_bert_v2"],
            path_format="{dataset_name}_{model_name}",
        )
        report_results(
            args,
            imbalance_factors=[None],
            dataset_names=["iwildcam"],
            model_names=["resnet50", "swin_large", "vit_large", "vit_large_384"],
            path_format="{dataset_name}_{model_name}",
        )


if __name__ == "__main__":
    main()
