# pylint: disable=missing-docstring
# pylint: disable=invalid-name
# pylint: disable=too-many-arguments
import numpy as np
from abstention.label_shift import AbstractImbalanceAdapter, PriorShiftAdapterFunc


def e_s_w_given_x(wVec, predY, pwr):
    wVec_pwr = wVec**pwr
    return np.matmul(predY, wVec_pwr)


def compute_b_w(wVec, predY, piVal):
    es_w1 = e_s_w_given_x(wVec, predY, pwr=1)
    es_w2 = e_s_w_given_x(wVec, predY, pwr=2)
    return 1 / (es_w2 / piVal + es_w1 / (1 - piVal)) * predY


def compute_b_w_constraint(wVec, scaledPredY, predY, piVal):
    es_w1 = e_s_w_given_x(wVec, predY, pwr=1)
    es_w2 = e_s_w_given_x(wVec, predY, pwr=2)
    return 1 / (es_w2 / piVal + es_w1 / (1 - piVal)) * scaledPredY


class EEImbalanceAdapter(AbstractImbalanceAdapter):
    def __init__(
        self,
        calibrator_factory=None,
        constraint=True,
        tolerance=1e-6,
        max_iterations=10,
        verbose=False,
    ):
        self.calibrator_factory = calibrator_factory
        self.constraint = constraint
        self.verbose = verbose
        self.tolerance = tolerance
        self.max_iterations = max_iterations

    def compute_weight(self, valid_labels, predYs, predYt, piVal):

        r = np.sum(predYt, axis=0).reshape(-1, 1)
        L = np.matmul(predYs.T, valid_labels)
        weight = piVal / (1 - piVal) * np.matmul(np.linalg.pinv(L), r)
        weight[weight < 0] = 0

        for _ in range(self.max_iterations):
            # weight = [w if w > 0 else 0 for w in weight]
            # weight = np.array(weight).reshape(-1, 1)
            prev_weight = weight
            # Labeled Data
            beff1 = compute_b_w(weight, predYs, piVal)
            L = np.matmul(beff1.T, valid_labels)
            # Unlabeled Data
            beff2 = compute_b_w(weight, predYt, piVal)
            r = np.sum(beff2, axis=0).reshape(-1, 1)
            weight = piVal / (1 - piVal) * np.matmul(np.linalg.pinv(L), r)
            weight[weight < 0] = 0
            if not np.sum(np.abs(prev_weight - weight) > self.tolerance):
                break

        return weight.reshape(-1)

    def compute_constraint_weight(self, valid_labels, predYs, predYt, piVal):
        # initialization
        n, k = valid_labels.shape
        p_s = np.sum(valid_labels, axis=0) / n
        p_s[p_s < 0] = 10**-6
        scale_py_k = np.array([-1 * p_s[i] / p_s[-1] for i in range(k - 1)])

        scaled_predYs = predYs[:, : k - 1] - p_s[: k - 1]
        scaled_predYt = predYt[:, : k - 1] - p_s[: k - 1]

        A = np.copy(valid_labels[:, :-1])
        A[np.where(valid_labels[:, -1] == 1)] = scale_py_k
        v = np.zeros((n, 1))
        v[np.where(valid_labels[:, -1] == 1)] = 1 / p_s[-1]

        # estimate the initial weights based on BBSE
        r = 1 / (1 - piVal) * np.sum(scaled_predYt, axis=0).reshape(
            -1, 1
        ) - 1 / piVal * np.matmul(scaled_predYs.T, v)
        L = 1 / piVal * np.matmul(scaled_predYs.T, A)
        weight = np.matmul(np.linalg.pinv(L), r)
        w_k = (1 - np.matmul(p_s[:-1], weight)) / p_s[-1]
        weight = np.concatenate((weight, [w_k]), axis=0)
        weight[weight < 0] = 0

        # update the weights based with iterations
        for _ in range(self.max_iterations):
            prev_weight = weight
            # Labeled Data
            beff1 = compute_b_w_constraint(weight, scaled_predYs, predYs, piVal)
            # Unlabeled Data
            beff2 = compute_b_w_constraint(weight, scaled_predYt, predYt, piVal)

            L = 1 / piVal * np.matmul(beff1.T, A)
            r = 1 / (1 - piVal) * np.sum(beff2, axis=0).reshape(
                -1, 1
            ) - 1 / piVal * np.matmul(beff1.T, v)

            weight = np.matmul(np.linalg.pinv(L), r)
            w_k = (1 - np.matmul(p_s[:-1], weight)) / p_s[-1]
            weight = np.concatenate((weight, [w_k]), axis=0)
            weight[weight < 0] = 0
            if not np.sum(np.abs(prev_weight - weight) > self.tolerance):
                break

        return weight.reshape(-1)

    def __call__(
        self, valid_labels, tofit_initial_posterior_probs, valid_posterior_probs
    ):

        piVal = valid_posterior_probs.shape[0] / (
            valid_posterior_probs.shape[0] + tofit_initial_posterior_probs.shape[0]
        )

        if self.calibrator_factory is not None:
            calibrator_func = self.calibrator_factory(
                valid_preacts=valid_posterior_probs,
                valid_labels=valid_labels,
                posterior_supplied=True,
            )
        else:
            calibrator_func = lambda x: x  # noqa: E731

        valid_posterior_probs = calibrator_func(valid_posterior_probs)
        tofit_initial_posterior_probs = calibrator_func(tofit_initial_posterior_probs)

        if not self.constraint:

            hat_weight = self.compute_weight(
                valid_labels=valid_labels,
                predYs=valid_posterior_probs,
                predYt=tofit_initial_posterior_probs,
                piVal=piVal,
            )

        else:
            hat_weight = self.compute_constraint_weight(
                valid_labels=valid_labels,
                predYs=valid_posterior_probs,
                predYt=tofit_initial_posterior_probs,
                piVal=piVal,
            )

        return PriorShiftAdapterFunc(
            multipliers=hat_weight, calibrator_func=calibrator_func
        )
