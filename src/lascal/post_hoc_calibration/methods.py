import math
from functools import partial
from itertools import product

import numpy as np
import torch
from joblib import Parallel, delayed
from scipy import optimize, special
from scipy.stats import multivariate_normal
from torch import nn
from torch.nn import functional as F
from tqdm.auto import tqdm

from lascal.post_hoc_calibration.prob_match_utils import (
    AbstractImbalanceAdapter,
    PriorShiftAdapterFunc,
    func_lp_source,
    func_lp_target,
)


class HeadToTail(nn.Module):
    def __init__(
        self,
        num_classes,
        features,
        logits,
        labels,
        train_features,
        train_labels,
        alpha=0.9,
        verbose=False,
    ):
        super(HeadToTail, self).__init__()
        num_head_cls = math.ceil(num_classes * 0.2)
        head_mask = np.array([-100] * num_classes)
        head_mask[:num_head_cls] = 1
        self.w = np.zeros((logits.shape[0]))

        dst_means, dst_covs = [], []
        target_norms = []
        for i in range(logits.shape[-1]):
            t_datas = train_features[train_labels == i, :]
            dst_means.append(np.mean(t_datas, axis=0))
            var = np.var(t_datas, axis=0)
            var[var < 1e-6] = 1e-6
            dst_covs.append(var)
            target_norms.append(
                multivariate_normal(
                    mean=dst_means[-1], cov=dst_covs[-1], allow_singular=False
                )
            )

        self.wasser_matrix = np.zeros((logits.shape[-1], logits.shape[-1]))
        for i in tqdm(range(logits.shape[-1]), disable=not verbose):
            for j in range(logits.shape[-1]):
                if i == j:
                    self.wasser_matrix[i, j] = -1e9
                elif head_mask[j] == -100:
                    self.wasser_matrix[i, j] = -1e9
                else:
                    self.wasser_matrix[i, j] = -(
                        self.wasserstein(
                            dst_means[i], dst_covs[i], dst_means[j], dst_covs[j]
                        )
                    ) / (train_features.shape[-1] ** (1 / 2))

            self.wasser_matrix[i] = special.softmax(self.wasser_matrix[i])

        for i in tqdm(range(logits.shape[0]), disable=not verbose):
            gt_cls = labels[i]
            if head_mask[gt_cls] == 1.0:
                self.w[i] = 1.0
            else:
                shift_mean = (
                    np.sum(
                        np.array(dst_means) * self.wasser_matrix[gt_cls][:, None],
                        axis=0,
                    )
                    * (1 - alpha)
                    + dst_means[gt_cls] * alpha
                )
                shift_cov = (
                    np.sum(
                        np.sqrt(np.array(dst_covs))
                        * self.wasser_matrix[gt_cls][:, None],
                        axis=0,
                    )
                    * (1 - alpha)
                    + np.sqrt(dst_covs[gt_cls]) * alpha
                ) ** 2
                self.w[i] = np.exp(
                    multivariate_normal(
                        mean=shift_mean, cov=shift_cov, allow_singular=False
                    ).logpdf(features[i])
                    - target_norms[gt_cls].logpdf(features[i])
                )

                self.w[i] = np.clip(self.w[i], 0.3, 5)

    def wasserstein(self, mu1, sigma1, mu2, sigma2):
        p1 = np.sum(np.power((mu1 - mu2), 2))
        p2 = np.sum(np.power(np.power(sigma1, 1 / 2) - np.power(sigma2, 1 / 2), 2))
        return p1 + p2

    def find_best_T(self, logit, label):
        one_hot_labels = np.eye(logit.shape[1])[label].astype(float)
        bnds = ((0.05, 5.0),)
        t = optimize.minimize(
            self.ll_t_da,
            1.0,
            args=(logit, one_hot_labels, self.w),
            method="L-BFGS-B",
            bounds=bnds,
            tol=1e-12,
            options={"disp": False},
        )

        return t.x

    def ll_t_da(self, t, *args):
        logits, labels, w = args
        logits = logits / t
        n = np.sum(np.clip(np.exp(logits), -1e20, 1e20), 1)
        p = np.clip(np.clip(np.exp(logits), -1e20, 1e20) / n[:, None], 1e-20, 1 - 1e-20)
        N = p.shape[0]
        ce = -np.sum(labels * np.log(p) * w[:, None]) / N
        return ce


class Cpcs(nn.Module):
    def __init__(self):
        super(Cpcs, self).__init__()

    def find_best_T(self, logits, labels, weight=None):
        def eval(x):
            x = torch.from_numpy(x)
            scaled_logits = logits.float() / x
            softmaxes = F.softmax(scaled_logits, dim=-1)
            ## Transform to onehot encoded labels
            labels_onehot = torch.FloatTensor(
                scaled_logits.shape[0], scaled_logits.shape[1]
            )
            labels_onehot.zero_()
            labels_onehot.scatter_(1, labels.long().view(len(labels), 1), 1)
            brier_score = torch.sum(
                (softmaxes - labels_onehot) ** 2, dim=1, keepdim=True
            )
            loss = torch.mean(brier_score * weight)
            return loss

        optimal_parameter = optimize.fmin(eval, 2.0, disp=False)

        return optimal_parameter[0]


class TransCal(nn.Module):
    def __init__(self, bias_term=True, variance_term=True):
        super(TransCal, self).__init__()
        self.bias_term = bias_term
        self.variance_term = variance_term

    def find_best_T(self, logits, weight, error, source_confidence):
        def eval(x):
            "x[0] ==> temperature T"
            scaled_logits = logits / x[0]

            "x[1] ==> learnable meta parameter \lambda"
            if self.bias_term:
                controled_weight = weight ** x[1]
            else:
                controled_weight = weight

            ## 1. confidence
            max_L = np.max(scaled_logits, axis=1, keepdims=True)
            exp_L = np.exp(scaled_logits - max_L)
            softmaxes = exp_L / np.sum(exp_L, axis=1, keepdims=True)
            confidences = np.max(softmaxes, axis=1)
            confidence = np.mean(confidences)

            ## 2. accuracy
            if self.variance_term:
                weighted_error = controled_weight * error
                cov_1 = np.cov(
                    np.concatenate((weighted_error, controled_weight), axis=1),
                    rowvar=False,
                )[0][1]
                var_w = np.var(controled_weight, ddof=1)
                eta_1 = -cov_1 / (var_w)

                cv_weighted_error = weighted_error + eta_1 * (controled_weight - 1)
                correctness = 1 - error
                cov_2 = np.cov(
                    np.concatenate((cv_weighted_error, correctness), axis=1),
                    rowvar=False,
                )[0][1]
                var_r = np.var(correctness, ddof=1)
                eta_2 = -cov_2 / (var_r)

                target_risk = (
                    np.mean(weighted_error)
                    + eta_1 * np.mean(controled_weight)
                    - eta_1
                    + eta_2 * np.mean(correctness)
                    - eta_2 * source_confidence
                )
                estimated_acc = 1.0 - target_risk
            else:
                weighted_error = controled_weight * error
                target_risk = np.mean(weighted_error)
                estimated_acc = 1.0 - target_risk

            ## 3. ECE on bin_size = 1 for optimizing.
            ## Note that: We still utilize a bin_size of 15 while evaluating,
            ## following the protocal of Guo et al. (On Calibration of Modern Neural Networks)
            loss = np.abs(confidence - estimated_acc)
            return loss

        bnds = ((1.0, None), (0.0, 1.0))
        optimal_parameter = optimize.minimize(
            eval, np.array([2.0, 0.5]), method="SLSQP", bounds=bnds
        )

        return optimal_parameter.x[0].item()


class Ets(nn.Module):

    def __init__(self):
        super(Ets, self).__init__()

    def ll_t(self, t, *args):
        # https://github.com/JiahaoChen1/Calibration/blob/main/calibration_method.py#L28
        logit, label = args
        # find optimal temperature with Cross-Entropy loss function
        logit = logit / t
        n = np.sum(np.clip(np.exp(logit), -1e20, 1e20), 1)
        p = np.clip(np.clip(np.exp(logit), -1e20, 1e20) / n[:, None], 1e-20, 1 - 1e-20)
        N = p.shape[0]
        ce = -np.sum(label * np.log(p)) / N
        return ce

    def ll_w(self, w, *args):
        # https://github.com/JiahaoChen1/Calibration/blob/main/calibration_method.py#L50
        # find optimal weight coefficients with Cros-Entropy loss function
        p0, p1, p2, label = args
        p = w[0] * p0 + w[1] * p1 + w[2] * p2
        N = p.shape[0]
        ce = -np.sum(label * np.log(p)) / N

        return ce

    # Ftting Temperature Scaling
    def train_temperature_scaling(self, logit, label):
        # https://github.com/JiahaoChen1/Calibration/blob/main/calibration_method.py#L108
        bnds = ((0.05, 5.0),)
        t = optimize.minimize(
            self.ll_t,
            1.0,
            args=(logit, label),
            method="L-BFGS-B",
            bounds=bnds,
            tol=1e-12,
            options={"disp": False},
        )

        return t.x

    def train_ensemble_temperature_scaling(self, logit, label, n_class):
        t = self.train_temperature_scaling(logit, label)
        w = self.util_ensemble_temperature_scaling(logit, label, t, n_class)

        return (t, w)

    def util_ensemble_temperature_scaling(self, logit, label, t, n_class):
        # https://github.com/JiahaoChen1/Calibration/blob/main/calibration_method.py#L133
        p1 = special.softmax(logit, axis=1)
        logit = logit / t
        p0 = special.softmax(logit, axis=1)
        p2 = np.ones_like(p0) / n_class

        bnds_w = (
            (0.0, 1.0),
            (0.0, 1.0),
            (0.0, 1.0),
        )

        def my_constraint_fun(x):
            return np.sum(x) - 1

        constraints = {
            "type": "eq",
            "fun": my_constraint_fun,
        }
        w = optimize.minimize(
            self.ll_w,
            (1.0, 0.0, 0.0),
            args=(p0, p1, p2, label),
            method="SLSQP",
            constraints=constraints,
            bounds=bnds_w,
            tol=1e-12,
            options={"disp": False},
        )
        w = w.x

        return w

    # Calibration: Ensemble Temperature Scaling
    def calibrate_ensemble_temperature_scaling(self, logits, t, w, n_class):
        # https://github.com/JiahaoChen1/Calibration/blob/main/calibration_method.py#L313
        p1 = special.softmax(logits, axis=1)
        logits = logits / t
        p0 = special.softmax(logits, axis=1)
        p2 = np.ones_like(p0) / n_class
        p = w[0] * p0 + w[1] * p1 + w[2] * p2
        p = np.log(np.clip(p, 1e-20, 1 - 1e-20))

        return p


class VectorScalingModel(nn.Module):
    def __init__(self, num_classes: int):
        super(VectorScalingModel, self).__init__()
        self.W = nn.Parameter(torch.ones(num_classes))
        self.b = nn.Parameter(torch.zeros(num_classes))

    def forward(self, logits):
        out = logits * self.W + self.b
        return out


class WenImbalanceAdapter15B(AbstractImbalanceAdapter):
    def __init__(self, calibrator_factory=None, verbose=False):
        self.calibrator_factory = calibrator_factory
        self.verbose = verbose

    def __call__(
        self, valid_labels, tofit_initial_posterior_probs, valid_posterior_probs
    ):

        if self.calibrator_factory is not None:
            calibrator_func = self.calibrator_factory(
                valid_preacts=valid_posterior_probs,
                valid_labels=valid_labels,
                posterior_supplied=True,
            )
        else:
            calibrator_func = lambda x: x

        tofit_initial_posterior_probs = calibrator_func(tofit_initial_posterior_probs)

        source_labels_count = valid_labels.sum(axis=0)
        prob_classwise_source = source_labels_count / source_labels_count.sum()
        num_classes = source_labels_count.shape[0]

        def func_smo(seed, prob_classwise_source, tofit_initial_posterior_probs):
            num_classes = tofit_initial_posterior_probs.shape[1]
            w = np.ones(shape=num_classes)
            repeat_times = 1000
            rng = np.random.RandomState(seed)
            x0_seq = []
            for i in range(repeat_times):
                w_new = w.copy()
                # randomly choose two dimension rdm & rdl
                probm = w / w.sum()
                probm[probm < 0] = 0
                rdm = rng.choice(num_classes, size=1, p=probm)[0]
                probl = probm.copy()
                probl[rdm] = 0
                probl[np.isclose(w, 0)] = 0
                probl += 1.0 / num_classes
                probl = probl / probl.sum()
                rdl = rng.choice(num_classes, size=1, p=probl)[0]
                # calculate Pi_P and Pi_Q w.r.t. w
                pip = prob_classwise_source
                # calculate Ai and Bi
                Ai = (
                    tofit_initial_posterior_probs[:, rdl]
                    - pip[rdl] / pip[rdm] * tofit_initial_posterior_probs[:, rdm]
                )
                Bi = np.zeros(shape=tofit_initial_posterior_probs.shape[0])
                for j in range(num_classes):
                    if j not in [rdl, rdm]:
                        Bi += w[j] * tofit_initial_posterior_probs[:, j]
                Bi += (
                    tofit_initial_posterior_probs[:, rdm]
                    * (w[rdm] * pip[rdm] + w[rdl] * pip[rdl])
                    / pip[rdm]
                )

                # THIS CAN BE REPLACED BY SCIPY.OPTIMIZE
                def func_inner(w_rdl, tofit_initial_posterior_probs, pip, Ai, Bi):
                    num_classes = tofit_initial_posterior_probs.shape[1]
                    C = 0
                    for k in range(num_classes):
                        C_tmp = (
                            np.mean(
                                (tofit_initial_posterior_probs[:, k] / pip[k])
                                / (Ai * w_rdl + Bi)
                            )
                            - 1
                        )
                        C += np.square(C_tmp)
                    return C

                equation = partial(
                    func_inner,
                    tofit_initial_posterior_probs=tofit_initial_posterior_probs,
                    pip=pip,
                    Ai=Ai,
                    Bi=Bi,
                )
                method = "Nelder-Mead"
                sol = optimize.minimize(equation, x0=w_new[rdl], method=method)
                w_new[rdl] = sol.x[0]
                # calculate range of w_rdl
                upper = (w[rdm] * pip[rdm] + w[rdl] * pip[rdl]) / pip[rdl]
                # truncate and update new w_rdl
                w_new[rdl] = max(0, min(w_new[rdl], upper))
                # update w_rdm
                w_new[rdm] = (
                    w[rdm] * pip[rdm] + w[rdl] * pip[rdl] - w_new[rdl] * pip[rdl]
                ) / pip[rdm]
                # next iteration
                w = w_new.copy()
                if i in [200 - 1, 500 - 1, 1000 - 1]:
                    x0_seq.append(w_new.copy())
            return x0_seq

        size = 8
        np.random.seed(101)
        random_seed_seq = np.random.randint(0, np.iinfo(np.uint32).max, size=size)
        n_jobs = 24

        results = Parallel(n_jobs=n_jobs, prefer="processes")(
            delayed(func_smo)(
                random_seed_seq[idx],
                prob_classwise_source,
                tofit_initial_posterior_probs,
            )
            for idx in range(size)
        )

        x0_seq = []
        for res in results:
            x0_seq += res

        def func(weight, source_labels_count, prob_target, sum_up=False):
            lp_source = func_lp_source(weight, source_labels_count)
            lp_target = func_lp_target(weight, prob_target)
            dis = np.square(lp_source - lp_target)
            return dis.sum() if sum_up else dis

        equation = partial(
            func,
            source_labels_count=source_labels_count,
            prob_target=tofit_initial_posterior_probs,
            sum_up=True,
        )

        good_sol = []
        num_x0 = len(x0_seq)
        method_seq = ["Nelder-Mead"]

        def run(method, x0):
            try:
                sol = optimize.minimize(equation, x0, method=method)
                x = sol.x if not sol.x.shape else sol.x[0]
                return [method, sol, x]
            except Exception:
                return [method, None, np.inf]

        n_jobs = 24
        opt_res = Parallel(n_jobs=n_jobs, prefer="processes")(
            delayed(run)(method, x0_seq[x0_idx])
            for method, x0_idx in product(method_seq, np.arange(num_x0))
        )

        good_sol = []
        for line in opt_res:
            method, sol, x = line
            good_sol.append(sol.x)
        for x0 in x0_seq:
            good_sol.append(x0)

        if len(good_sol) == 0:
            raise Exception("Solve Equation failed!")

        gap_seq = []
        for x in good_sol:
            x_clip = x.copy()
            x_clip[x_clip <= 0] = 0
            lp_source = func_lp_source(x_clip, source_labels_count)
            lp_target = func_lp_target(x_clip, tofit_initial_posterior_probs)
            gap = np.square(lp_source - lp_target).sum()
            gap_seq.append(gap)
        gap_seq = np.array(gap_seq)

        # newly chanaged 1224
        tmp = tofit_initial_posterior_probs.mean(axis=0)
        tmpsort = np.argsort(tmp)

        ccc = []
        for i in range(len(good_sol)):
            best_x = good_sol[i]
            best_x[best_x <= 0] = 0
            ccc.append(np.square(tmp - best_x).mean())
        ccc = np.array(ccc)

        sortidx = np.argsort(gap_seq)
        for best_idx in sortidx[1:]:
            best_x = good_sol[best_idx]
            best_x[best_x <= 0] = 0
            if ccc[best_idx] < np.median(ccc) + 3 * np.std(ccc):
                break

        prob_classwise_target_hat = func_lp_source(best_x, source_labels_count)
        weights = prob_classwise_target_hat / prob_classwise_source
        weights = 1.0 * (weights * (weights >= 0))

        return PriorShiftAdapterFunc(
            multipliers=weights, calibrator_func=calibrator_func
        )
