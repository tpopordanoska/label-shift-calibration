from os.path import join as pjoin

import numpy as np
import torch
from abstention.calibration import TempScaling
from abstention.label_shift import EMImbalanceAdapter
from scipy import special
from sklearn import linear_model
from sklearn.isotonic import IsotonicRegression
from torch import nn, optim
from tqdm.auto import tqdm

from lascal import Ece, EceLabelShift, get_importance_weights
from lascal.post_hoc_calibration.methods import (
    Cpcs,
    Ets,
    HeadToTail,
    TransCal,
    VectorScalingModel,
    WenImbalanceAdapter15B,
)
from lascal.utils import SoftmaxClipper, initialize_overwatch
from lascal.utils.constants import MAX_TEMP, MIN_TEMP, STEPS

# Initialize Overwatch =>> Wraps `logging.Logger`
overwatch = initialize_overwatch(__name__)


def cal_acc_error(logit, label):
    # https://github.com/thuml/TransCal/blob/master/pytorch/3_TransCal.py
    softmaxes = nn.Softmax(dim=1)(logit)
    confidences, predictions = torch.max(softmaxes, 1)
    accuracies = predictions.eq(label)
    accuracy = accuracies.float().mean()
    confidence = confidences.float().mean()
    error = 1 - accuracies.float()
    error = error.view(len(error), 1).float().numpy()
    return accuracy, confidence, error


def get_weight_feature_space(train_feature, target_feature, source_feature, **kwargs):
    source_indices = kwargs.pop("source_indices", None)
    if source_indices:
        source_feature = source_feature[source_indices]
    # https://github.com/thuml/TransCal/blob/master/pytorch/3_TransCal.py
    n_tr, _ = train_feature.shape
    n_t, _ = target_feature.shape
    if n_tr < n_t:
        sample_index = np.random.choice(n_tr, n_t, replace=True)
        train_feature = train_feature[sample_index]
        sample_num = n_t
    elif n_tr > n_t:
        sample_index = np.random.choice(n_t, n_tr, replace=True)
        target_feature = target_feature[sample_index]
        sample_num = n_tr

    combine_feature = np.concatenate((train_feature, target_feature))
    combine_label = np.asarray([1] * sample_num + [0] * sample_num, dtype=np.int32)
    domain_classifier = linear_model.LogisticRegression()
    domain_classifier.fit(combine_feature, combine_label)
    domain_out = domain_classifier.predict_proba(source_feature)
    weight = domain_out[:, :1] / domain_out[:, 1:]

    return weight


class Calibrator:

    def __init__(
        self,
        experiment_path: str = None,
        verbose: bool = False,
        covariate: bool = False,
        criterion: str = "cross_entropy",
    ):
        self.experiment_path = experiment_path
        self.verbose = verbose
        self.target_name = "target" if not covariate else "target_cov"
        assert criterion in ["cross_entropy", "ece"], f"Invalid: {criterion}"
        self.criterion = Ece() if criterion == "ece" else nn.CrossEntropyLoss()
        self.softmax_clipper = SoftmaxClipper()

    def calibrate(self, method_name: str, source_agg, target_agg, train_agg, **kwargs):
        # Requires a method name to calibrate with, and source, target, and train aggregates
        name2method = {
            "uncal": self.uncal,
            "head_to_tail": self.head_to_tail,
            "ensemble_temp_scale": self.ensemble_temp_scale,
            "isotonic_regression": self.isotonic_regression,
            "cpcs": self.cpcs,
            "transcal": self.transcal,
            "em_alexandari": self.em_alexandari,
            "probability_matching": self.probability_matching,
            "vector_scale_source": self.vector_scale_source,
            "vector_scale_target": self.vector_scale_target,
            "temp_scale_source": self.temperature_scale_source,
            "temp_scale_target": self.temperature_scale_target,
            "lascal": self.lascal,
        }
        return name2method[method_name](
            source_agg=source_agg, target_agg=target_agg, train_agg=train_agg, **kwargs
        )

    def uncal(self, **kwargs):
        source_agg = kwargs.pop("source_agg")
        target_agg = kwargs.pop("target_agg")

        return {
            "source": {
                "y_logits": source_agg["y_logits"],
                "y_true": source_agg["y_true"],
            },
            "target": {
                "y_logits": target_agg["y_logits"],
                "y_true": target_agg["y_true"],
            },
        }

    def head_to_tail(self, **kwargs):
        assert self.experiment_path, "Experiment path is required for HeadToTail"
        # Get source and target
        train_agg = kwargs.pop("train_agg")
        source_agg = kwargs.pop("source_agg")
        target_agg = kwargs.pop("target_agg")
        # Get features
        features = {
            feature_name: torch.load(
                pjoin(self.experiment_path, f"{feature_name}_features.pt")
            )
            for feature_name in ["train", "target", "source"]
        }
        num_classes = source_agg["y_true"].max().item() + 1
        head_to_tail = HeadToTail(
            num_classes=num_classes,
            features=features["source"].numpy(),
            logits=source_agg["y_logits"].numpy(),
            labels=source_agg["y_true"].numpy(),
            train_features=features["train"].numpy(),
            train_labels=train_agg["y_true"].numpy(),
            verbose=self.verbose,
        )
        optim_temp = head_to_tail.find_best_T(
            source_agg["y_logits"].numpy(), source_agg["y_true"].numpy()
        )

        if self.verbose:
            overwatch.info(f"Temperature found with HeadToTail is: {optim_temp[0]}")

        return {
            "source": {
                "y_logits": source_agg["y_logits"],
                "y_true": source_agg["y_true"],
            },
            "target": {
                "y_logits": target_agg["y_logits"] / optim_temp,
                "y_true": target_agg["y_true"],
            },
        }

    def ensemble_temp_scale(self, **kwargs):
        source_agg = kwargs.pop("source_agg")
        target_agg = kwargs.pop("target_agg")

        n_class = source_agg["y_true"].max().item() + 1
        one_hot_labels = np.eye(source_agg["y_logits"].size(1))[
            source_agg["y_true"].numpy()
        ].astype(float)
        t, w = Ets().train_ensemble_temperature_scaling(
            source_agg["y_logits"].numpy(), one_hot_labels, n_class=n_class
        )

        calibrated_probs = Ets().calibrate_ensemble_temperature_scaling(
            logits=target_agg["y_logits"].numpy(), t=t, w=w, n_class=n_class
        )
        calibrated_probs = torch.from_numpy(calibrated_probs)

        return {
            "source": {
                "y_logits": source_agg["y_logits"],
                "y_true": source_agg["y_true"],
            },
            "target": {
                "y_logits": calibrated_probs,
                "y_true": target_agg["y_true"],
            },
        }

    def isotonic_regression(self, **kwargs):
        source_agg = kwargs.pop("source_agg")
        target_agg = kwargs.pop("target_agg")

        # Convert to numpy
        for name in ["y_true", "y_logits"]:
            source_agg[name] = source_agg[name].numpy()
            target_agg[name] = target_agg[name].numpy()

        probabilities_source = special.softmax(source_agg["y_logits"], axis=-1)
        probabilities_target = special.softmax(target_agg["y_logits"], axis=-1)

        calibrated_scores_test = np.ones_like(probabilities_target)
        for class_idx in range(probabilities_target.shape[1]):
            iso_reg = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
            mask = source_agg["y_true"] == class_idx
            iso_reg.fit(probabilities_source[:, class_idx], mask)
            calibrated_scores_test[:, class_idx] = iso_reg.predict(
                probabilities_target[:, class_idx]
            )

        row_sums = np.sum(calibrated_scores_test, axis=1)
        calibrated_scores_test = calibrated_scores_test / row_sums[:, np.newaxis]
        calibrated_scores_test = torch.from_numpy(calibrated_scores_test)

        calibrated_logits_test = torch.log(calibrated_scores_test) + 10.0

        return {
            "source": {
                "y_logits": torch.from_numpy(source_agg["y_logits"]),
                "y_true": torch.from_numpy(source_agg["y_true"]),
            },
            "target": {
                "y_logits": calibrated_logits_test,
                "y_true": torch.from_numpy(target_agg["y_true"]),
            },
        }

    def cpcs(self, **kwargs):
        assert self.experiment_path, "Experiment path is required for CPCS"
        # Get source and target
        source_agg = kwargs.pop("source_agg")
        target_agg = kwargs.pop("target_agg")
        # Get features
        features = [
            torch.load(pjoin(self.experiment_path, f"{feature_name}_features.pt"))
            for feature_name in ["train", self.target_name, "source"]
        ]
        source_indices = (
            source_agg["source_indices"] if "source_indices" in source_agg else None
        )
        weight = get_weight_feature_space(*features, source_indices=source_indices)
        optim_temp = Cpcs().find_best_T(
            source_agg["y_logits"], source_agg["y_true"], weight=weight
        )

        if self.verbose:
            overwatch.info(f"Temperature found with CPCS is: {optim_temp}")

        return {
            "source": {
                "y_logits": source_agg["y_logits"],
                "y_true": source_agg["y_true"],
            },
            "target": {
                "y_logits": target_agg["y_logits"] / optim_temp,
                "y_true": target_agg["y_true"],
            },
        }

    def transcal(self, **kwargs):
        assert self.experiment_path, "Experiment path is required for TransCal"
        # Get source and target
        source_agg = kwargs.pop("source_agg")
        target_agg = kwargs.pop("target_agg")
        # Optimal temp on source
        optim_temp_source = self._temperature_scale(
            source_agg["y_logits"], source_agg["y_true"]
        )
        _, source_confidence, error_source_val = cal_acc_error(
            source_agg["y_logits"] / optim_temp_source, source_agg["y_true"]
        )
        # Obtain features and get weight
        features = [
            torch.load(pjoin(self.experiment_path, f"{feature_name}_features.pt"))
            for feature_name in ["train", self.target_name, "source"]
        ]
        source_indices = (
            source_agg["source_indices"] if "source_indices" in source_agg else None
        )
        weight = get_weight_feature_space(*features, source_indices=source_indices)
        # Find optimal temp with TransCal
        optim_temp = TransCal().find_best_T(
            target_agg["y_logits"].numpy(),
            weight,
            error_source_val,
            source_confidence.item(),
        )

        if self.verbose:
            overwatch.info(f"Temperature found with TransCal is: {optim_temp}")

        return {
            "source": {
                "y_logits": source_agg["y_logits"],
                "y_true": source_agg["y_true"],
            },
            "target": {
                "y_logits": target_agg["y_logits"] / optim_temp,
                "y_true": target_agg["y_true"],
            },
        }

    def _vector_scale(self, logits, labels):
        # Get calibration model and optimizer
        cal_model = VectorScalingModel(num_classes=logits.shape[1])
        optimizer = optim.SGD(cal_model.parameters(), lr=0.01, momentum=0.9)
        # Train calibration model
        for _ in range(1000):
            optimizer.zero_grad()
            scaled_logits = cal_model(logits)
            loss = self.criterion(scaled_logits, labels)
            loss.backward()
            optimizer.step()

        return cal_model

    def vector_scale_source(self, **kwargs):
        # Get source and target
        source_agg = kwargs.pop("source_agg")
        target_agg = kwargs.pop("target_agg")

        optim_model = self._vector_scale(
            logits=source_agg["y_logits"], labels=source_agg["y_true"]
        )
        scaled_target_logits = optim_model(target_agg["y_logits"])

        return {
            "source": {
                "y_logits": source_agg["y_logits"],
                "y_true": source_agg["y_true"],
            },
            "target": {
                "y_logits": scaled_target_logits,
                "y_true": target_agg["y_true"],
            },
        }

    def vector_scale_target(self, **kwargs):
        # Get target
        source_agg = kwargs.pop("source_agg")
        target_agg = kwargs.pop("target_agg")

        optim_model = self._vector_scale(
            logits=target_agg["y_logits"], labels=target_agg["y_true"]
        )

        scaled_target_logits = optim_model(target_agg["y_logits"])

        return {
            "source": {
                "y_logits": source_agg["y_logits"],
                "y_true": source_agg["y_true"],
            },
            "target": {
                "y_logits": scaled_target_logits,
                "y_true": target_agg["y_true"],
            },
        }

    def _temperature_scale(self, logits, labels):
        optim_temp = -1
        best_loss = torch.finfo(torch.float).max
        for temp in tqdm(
            torch.linspace(MIN_TEMP, MAX_TEMP, steps=STEPS), disable=not self.verbose
        ):
            loss = self.criterion((logits / temp), labels).mean()
            if loss < best_loss:
                best_loss = loss
                optim_temp = temp

        return optim_temp

    def temperature_scale_source(self, **kwargs):
        # Get source and target
        source_agg = kwargs.pop("source_agg")
        target_agg = kwargs.pop("target_agg")
        optim_temp = self._temperature_scale(
            logits=source_agg["y_logits"], labels=source_agg["y_true"]
        )

        if self.verbose:
            overwatch.info(f"Temperature found on Source is: {optim_temp.item()}")

        return {
            "source": {
                "y_logits": source_agg["y_logits"],
                "y_true": source_agg["y_true"],
            },
            "target": {
                "y_logits": target_agg["y_logits"] / optim_temp,
                "y_true": target_agg["y_true"],
            },
        }

    def temperature_scale_target(self, **kwargs):
        # Get source and target
        source_agg = kwargs.pop("source_agg")
        target_agg = kwargs.pop("target_agg")
        # Get optimal temperature
        optim_temp = self._temperature_scale(
            logits=target_agg["y_logits"], labels=target_agg["y_true"]
        )

        if self.verbose:
            overwatch.info(f"Temperature found on Target is: {optim_temp.item()}")

        return {
            "source": {
                "y_logits": source_agg["y_logits"],
                "y_true": source_agg["y_true"],
            },
            "target": {
                "y_logits": target_agg["y_logits"] / optim_temp,
                "y_true": target_agg["y_true"],
            },
        }

    @staticmethod
    def inv_softmax(x, c=torch.log(torch.tensor(10))):
        return torch.log(x) + c

    def em_alexandari(self, **kwargs):
        # Get source and target
        source_agg = kwargs.pop("source_agg")
        target_agg = kwargs.pop("target_agg")
        # Convert to one-hot
        num_classes = source_agg["y_logits"].size(1)
        y_source_ohe = np.eye(num_classes)[source_agg["y_true"].numpy()].astype(float)
        # As per Colab: https://colab.research.google.com/github/kundajelab/labelshiftexperiments/blob/master/notebooks/demo/blog_colab.ipynb#scrollTo=QA8EnUvcWOfd
        # Instantiate the BCTS calibrator factory and specify that we would like to use Maximum Likelihood (EM) for the
        # label shift adaptation, with BCTS for calibration
        imbalance_adapter = EMImbalanceAdapter(
            calibrator_factory=TempScaling(verbose=False, bias_positions="all")
        )
        # Get the function that will do the label shift adaptation (creating this
        # function requires supplying the validation set labels/predictions as well as
        # the test-set predictions)
        test_softmax_preds = target_agg["y_logits"].softmax(-1).numpy()
        val_softmax_preds = source_agg["y_logits"].softmax(-1).numpy()
        # Clip
        val_softmax_preds, test_softmax_preds = self.softmax_clipper(
            val_softmax_preds=val_softmax_preds,
            test_softmax_preds=test_softmax_preds,
            method="em-bcts",
        )
        # Adapt
        imbalance_adapter_func = imbalance_adapter(
            valid_labels=y_source_ohe,
            tofit_initial_posterior_probs=test_softmax_preds,
            valid_posterior_probs=val_softmax_preds,
        )
        adapted_test_softmax_preds = torch.from_numpy(
            imbalance_adapter_func(test_softmax_preds)
        )
        test_logits = self.inv_softmax(adapted_test_softmax_preds)

        return {
            "source": {
                "y_logits": source_agg["y_logits"],
                "y_true": source_agg["y_true"],
            },
            "target": {"y_logits": test_logits, "y_true": target_agg["y_true"]},
        }

    def probability_matching(self, **kwargs):
        # Get source and target
        source_agg = kwargs.pop("source_agg")
        target_agg = kwargs.pop("target_agg")
        # Convert to one-hot
        num_classes = source_agg["y_logits"].size(1)
        y_source_ohe = np.eye(num_classes)[source_agg["y_true"].numpy()].astype(float)
        imbalance_adapter = WenImbalanceAdapter15B(
            calibrator_factory=TempScaling(verbose=False, bias_positions="all")
        )
        # Get shifted test predictions and calibrate
        test_softmax_preds = target_agg["y_logits"].softmax(-1).numpy()
        val_softmax_preds = source_agg["y_logits"].softmax(-1).numpy()
        # Clip
        val_softmax_preds, test_softmax_preds = self.softmax_clipper(
            val_softmax_preds=val_softmax_preds,
            test_softmax_preds=test_softmax_preds,
            method="prob_match",
        )
        # Adapt
        imbalance_adapter_func = imbalance_adapter(
            valid_labels=y_source_ohe,
            tofit_initial_posterior_probs=test_softmax_preds,
            valid_posterior_probs=val_softmax_preds,
        )
        adapted_test_softmax_preds = torch.from_numpy(
            imbalance_adapter_func(test_softmax_preds)
        )
        test_logits = self.inv_softmax(adapted_test_softmax_preds)

        return {
            "source": {
                "y_logits": source_agg["y_logits"],
                "y_true": source_agg["y_true"],
            },
            "target": {"y_logits": test_logits, "y_true": target_agg["y_true"]},
        }

    def lascal(self, **kwargs):
        source_agg = kwargs.pop("source_agg")
        target_agg = kwargs.pop("target_agg")
        weights_method = kwargs.pop("weights_method", "rlls-hard")
        p = kwargs.pop("p", 2)
        classwise = kwargs.pop("classwise", True)

        ece_criterion = EceLabelShift(
            p=p, n_bins=15, adaptive_bins=True, classwise=classwise
        )
        optim_temp = -1
        best_loss = torch.finfo(torch.float).max

        for temp in tqdm(
            torch.linspace(MIN_TEMP, MAX_TEMP, steps=STEPS), disable=not self.verbose
        ):
            # Prepare source labels
            num_classes = source_agg["y_logits"].size(1)
            y_source_ohe = np.eye(num_classes)[source_agg["y_true"].numpy()].astype(
                float
            )
            scaled_logits_source = source_agg["y_logits"] / temp
            scaled_logits_target = target_agg["y_logits"] / temp
            # Get weight
            output = get_importance_weights(
                valid_preds=scaled_logits_source.softmax(-1).numpy(),
                valid_labels=y_source_ohe,
                shifted_test_preds=scaled_logits_target.softmax(-1).numpy(),
                method=weights_method,
            )
            # Measure loss
            loss = ece_criterion(
                logits_source=scaled_logits_source,
                labels_source=source_agg["y_true"],
                logits=scaled_logits_target,
                weights=output["weights"],
            ).mean()
            if loss < best_loss:
                best_loss = loss
                optim_temp = temp

        if self.verbose:
            overwatch.info(f"Temperature found with LasCal is: {optim_temp.item()}")

        return {
            "source": {
                "y_logits": source_agg["y_logits"],
                "y_true": source_agg["y_true"],
            },
            "target": {
                "y_logits": target_agg["y_logits"] / optim_temp,
                "y_true": target_agg["y_true"],
            },
        }
