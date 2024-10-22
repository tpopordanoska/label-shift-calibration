import numpy as np
import torch

from lascal import Ece, EceLabelShift


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

    def __call__(self, **kwargs):
        # First, let's check if the estimator is Ece or EceLabelShift
        if isinstance(self.estimator, Ece):
            assert "logits" in kwargs and "labels" in kwargs
            logits = kwargs["logits"]
            labels = kwargs["labels"]
            logits_source, labels_source, weights = None, None, None
        elif isinstance(self.estimator, EceLabelShift):
            for key in ["logits", "logits_source", "labels_source", "weights"]:
                assert key in kwargs
            logits = kwargs["logits"]
            logits_source = kwargs["logits_source"]
            labels_source = kwargs["labels_source"]
            weights = kwargs["weights"]
            labels = None
        
        # Calculate bootstrap estimates
        bootstrap_estimates = torch.zeros(self.num_bootstrap_samples)
        for i in range(self.num_bootstrap_samples):
            # Resample with replacement
            bootstrap_logits_target, bootstrap_labels_target = (
                self.get_bootstrap_sample(logits, labels)
            )
            estimate = self.estimator(
                logits=bootstrap_logits_target,
                labels=bootstrap_labels_target,
                logits_source=logits_source,
                labels_source=labels_source,
                weigths=weights
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
