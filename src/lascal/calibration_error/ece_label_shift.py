import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class EceLabelShift(nn.Module):
    """
    Compute ECE (Expected Calibration Error) under label shift
    """

    def __init__(self, p=2, n_bins=15, adaptive_bins=True, classwise=False):
        super(EceLabelShift, self).__init__()
        self.p = p
        self.n_bins = n_bins
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

    def get_ece(self, confidences_source, confidences_target, labels_source, weight):
        tmp_sum = torch.zeros(1, device=confidences_source.device)
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            in_bin_source = confidences_source.gt(
                bin_lower.item()
            ) * confidences_source.le(bin_upper.item())
            in_bin_target = confidences_target.gt(
                bin_lower.item()
            ) * confidences_target.le(bin_upper.item())
            if in_bin_target.sum() > 1:
                normalizer = (len(confidences_target) - 1) / len(confidences_source)
                weighted_num = weight * labels_source[in_bin_source].float().sum()
                cond_expect = normalizer * weighted_num / (in_bin_target.sum() - 1)
                for point_target in confidences_target[in_bin_target]:
                    tmp_sum += torch.abs(point_target - cond_expect) ** self.p

        return tmp_sum / len(confidences_target)

    def get_ece_top_label(
        self,
        confidences_source,
        confidences_target,
        labels_source,
        predictions_source,
        weights,
    ):
        tmp_sum = torch.zeros(1, device=confidences_source.device)
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            in_bin_source = confidences_source.gt(
                bin_lower.item()
            ) * confidences_source.le(bin_upper.item())
            in_bin_target = confidences_target.gt(
                bin_lower.item()
            ) * confidences_target.le(bin_upper.item())

            labels_source_in_bin = labels_source[in_bin_source]
            predictions_source_in_bin = predictions_source[in_bin_source]
            accuracies_source_in_bin = predictions_source_in_bin.eq(
                labels_source_in_bin
            )
            if in_bin_target.sum() > 1:
                normalizer = (len(confidences_target) - 1) / len(confidences_source)
                weighted_num = 0
                for c in range(len(weights)):
                    labels_source_in_bin_per_class_idx = labels_source_in_bin == c
                    class_mask_in_bin_per_class = accuracies_source_in_bin[
                        labels_source_in_bin_per_class_idx
                    ]
                    weighted_num += (
                        weights[c] * class_mask_in_bin_per_class.float().sum()
                    )

                cond_expect = normalizer * weighted_num / (in_bin_target.sum() - 1)
                for point_target in confidences_target[in_bin_target]:
                    tmp_sum += torch.abs(point_target - cond_expect) ** self.p

        return tmp_sum / len(confidences_target)

    def forward(self, logits_source, labels_source, logits_target, weight, **kwargs):
        softmaxes_source = F.softmax(logits_source, dim=1)
        softmaxes_target = F.softmax(logits_target, dim=1)
        if self.classwise:
            num_classes = softmaxes_source.shape[1]
            per_class_ce = None
            for i in range(num_classes):
                class_confidences_source = softmaxes_source[:, i]
                class_confidences_target = softmaxes_target[:, i]
                labels_in_source = labels_source.eq(i)
                self.set_bins(class_confidences_target)

                class_ece = self.get_ece(
                    class_confidences_source,
                    class_confidences_target,
                    labels_in_source,
                    weight[i],
                )
                if i == 0:
                    per_class_ce = class_ece
                else:
                    per_class_ce = torch.cat((per_class_ce, class_ece), dim=0)
            return per_class_ce

        else:
            # Top label:
            confidences_source, predictions_source = torch.max(softmaxes_source, 1)
            confidences_target, _ = torch.max(softmaxes_target, 1)
            self.set_bins(confidences_target)
            ece = self.get_ece_top_label(
                confidences_source,
                confidences_target,
                labels_source,
                predictions_source,
                weight,
            )

            return ece
