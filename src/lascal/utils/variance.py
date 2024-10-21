import numpy as np
import torch


class BootstrapMeanVarEstimator:
    def __init__(self, estimator, num_bootstrap_samples=100, reported="mean"):
        self.estimator = estimator
        self.num_bootstrap_samples = num_bootstrap_samples
        self.reported = reported

    def get_bootstrap_sample(self, logits, labels):
        sample_size = len(logits)
        indices = np.random.choice(sample_size, size=sample_size, replace=True)
        bootstrap_logits = logits[indices]
        bootstrap_labels = labels[indices]

        return bootstrap_logits, bootstrap_labels

    def __call__(
        self,
        logits_source,
        labels_source,
        logits_target,
        labels_target,
    ):
        bootstrap_estimates = torch.zeros(self.num_bootstrap_samples)
        for i in range(self.num_bootstrap_samples):
            # Resample with replacement
            bootstrap_logits_target, bootstrap_labels_target = (
                self.get_bootstrap_sample(logits_target, labels_target)
            )
            estimate = self.estimator(
                logits_source=logits_source,
                labels_source=labels_source,
                logits_target=bootstrap_logits_target,
                labels_target=bootstrap_labels_target,
            )
            if self.estimator.classwise:
                if self.reported == "mean":
                    estimate = estimate.nanmean()
                elif self.reported == "sum":
                    estimate = estimate.nansum()
            bootstrap_estimates[i] = estimate

        # Calculate mean and variance of the bootstrap estimates
        mean_estimate = torch.mean(bootstrap_estimates)
        variance_estimate = torch.var(bootstrap_estimates, unbiased=True)

        return mean_estimate.item(), variance_estimate.item()
