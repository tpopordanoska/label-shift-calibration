import numpy as np


def map_to_softmax_format_if_appropriate(values):
    if len(values.shape) == 1 or values.shape[1] == 1:
        values = np.squeeze(values)
        softmax_values = np.zeros((len(values), 2))
        softmax_values[:, 1] = values
        softmax_values[:, 0] = 1 - values
    else:
        softmax_values = values
    return softmax_values


def func_lp_source(weight, source_labels_count):
    lp_source = source_labels_count * weight
    lp_source = lp_source / np.sum(lp_source)
    return lp_source


def func_lp_target(weight, prob_target):
    weighted_prob = prob_target * weight
    weighted_prob = weighted_prob / weighted_prob.sum(axis=1, keepdims=True)
    lp_target = np.mean(weighted_prob, axis=0)
    return lp_target


class AbstractImbalanceAdapter(object):
    def __call__(
        self, valid_labels, tofit_initial_posterior_probs, valid_posterior_probs
    ):
        raise NotImplementedError()


class AbstractImbalanceAdapterFunc(object):
    def __call__(self, unadapted_posterior_probs):
        raise NotImplementedError()


class PriorShiftAdapterFunc(AbstractImbalanceAdapterFunc):
    def __init__(self, multipliers, calibrator_func=lambda x: x):
        self.multipliers = multipliers
        self.calibrator_func = calibrator_func

    def __call__(self, unadapted_posterior_probs):

        unadapted_posterior_probs = self.calibrator_func(unadapted_posterior_probs)

        # if supplied probs are in binary format, convert to softmax format
        softmax_unadapted_posterior_probs = map_to_softmax_format_if_appropriate(
            values=unadapted_posterior_probs
        )
        adapted_posterior_probs_unnorm = (
            softmax_unadapted_posterior_probs * self.multipliers[None, :]
        )
        adapted_posterior_probs = (
            adapted_posterior_probs_unnorm
            / np.sum(adapted_posterior_probs_unnorm, axis=-1)[:, None]
        )

        # return to binary format if appropriate
        if (
            len(unadapted_posterior_probs.shape) == 1
            or unadapted_posterior_probs.shape[1] == 1
        ):
            if len(unadapted_posterior_probs.shape) == 1:
                adapted_posterior_probs = adapted_posterior_probs[:, 1]
            else:
                if unadapted_posterior_probs.shape[1] == 1:
                    adapted_posterior_probs = adapted_posterior_probs[:, 1:2]

        return adapted_posterior_probs
