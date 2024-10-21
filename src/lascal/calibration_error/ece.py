from collections import defaultdict

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class Ece(nn.Module):
    """
    Compute ECE (Expected Calibration Error)
    """

    def __init__(
        self, p=2, n_bins=15, version="our", adaptive_bins=True, classwise=True
    ):
        super(Ece, self).__init__()
        self.p = p
        self.n_bins = n_bins
        self.version = version
        self.adaptive_bins = adaptive_bins
        self.classwise = classwise

    def set_bins(self, confidences):
        if self.adaptive_bins:
            _, bin_boundaries = np.histogram(
                confidences.cpu().detach(),
                self.histedges_equalN(confidences.cpu().detach()),
            )
        else:
            bin_boundaries = torch.linspace(0, 1, self.n_bins + 1)

        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]

    def histedges_equalN(self, x):
        npt = len(x)
        return np.interp(
            np.linspace(0, npt, self.n_bins + 1), np.arange(npt), np.sort(x)
        )

    def get_ece(self, confidences, labels):
        tmp_sum = torch.zeros(1, device=confidences.device)
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            if self.version == "our":
                if len(in_bin) > 1:
                    conf_in_bin = confidences[in_bin]
                    labels_in_bin = labels[in_bin].float()
                    # Subtract label from total sum because of i != j condition
                    cond_expects = (labels_in_bin.sum() - labels_in_bin) / (
                        len(labels_in_bin) - 1
                    )
                    tmp_sum += torch.abs(conf_in_bin - cond_expects).pow(self.p).sum()
            else:
                prop_in_bin = in_bin.float().mean()
                if prop_in_bin.item() > 0:
                    accuracy_in_bin = labels[in_bin].float().mean()
                    avg_confidence_in_bin = confidences[in_bin].mean()
                    tmp_sum += prop_in_bin * (
                        torch.abs(avg_confidence_in_bin - accuracy_in_bin) ** self.p
                    )

        return tmp_sum / len(confidences) if self.version == "our" else tmp_sum

    def forward(self, logits_target, labels_target, **kwargs):
        softmaxes = F.softmax(logits_target, dim=1)

        if self.classwise:
            num_classes = softmaxes.shape[1]
            per_class_ce = None
            for i in range(num_classes):
                class_confidences = softmaxes[:, i]
                labels_in_class = labels_target.eq(i)
                self.set_bins(class_confidences)

                class_ece = self.get_ece(class_confidences, labels_in_class)
                if i == 0:
                    per_class_ce = class_ece
                else:
                    per_class_ce = torch.cat((per_class_ce, class_ece), dim=0)
            return per_class_ce

        else:
            confidences, predictions = torch.max(softmaxes, 1)
            accuracies = predictions.eq(labels_target)
            self.set_bins(confidences)
            ece = self.get_ece(confidences, accuracies)

            return torch.round(ece, decimals=4)
