from collections import defaultdict

import numpy as np
import torch


class DatasetSubsampler:

    def __init__(self):
        self.dataset2subsampler = {
            "cifar10": self.subsample_cifar,
            "cifar100": self.subsample_cifar,
            "amazon": self.subsample_amazon,
            "iwildcam": self.subsample_iwildcam,
            "imagenet": self.subsample_imagenet,
            "tiny_imagenet": self.subsample_tiny_imagenet,
        }

    def subsample_imagenet(self, source_agg, target_agg):
        bincount = torch.bincount(source_agg["y_true"])
        # Keeping only classes for which source is reliable
        keep_classes = torch.where((bincount > 10) & (bincount < 50))[0]
        # Subsample target
        classid2indices = self.get_classid2indices(target_agg["y_true"])
        # Sample indices
        sampled_target_indices = []
        for class_idx in keep_classes.tolist():
            indices = classid2indices[class_idx]
            sampled_target_indices.extend(indices)
        # Subsample_target
        target_logits = target_agg["y_logits"][sampled_target_indices][:, keep_classes]
        target_labels = self.labels_remmaping(
            target_agg["y_true"][sampled_target_indices], keep_classes=keep_classes
        )
        target_agg = {"y_logits": target_logits, "y_true": target_labels}
        # Subsample source
        classid2indices = self.get_classid2indices(source_agg["y_true"])
        # Sample indices
        sampled_source_indices = []
        for class_idx in keep_classes.tolist():
            indices = classid2indices[class_idx]
            sampled_source_indices.extend(indices)

        source_logits = source_agg["y_logits"][sampled_source_indices][:, keep_classes]
        source_labels = self.labels_remmaping(
            source_agg["y_true"][sampled_source_indices], keep_classes=keep_classes
        )
        # Subsample source agg
        source_agg = {
            "y_logits": source_logits,
            "y_true": source_labels,
            "source_indices": sampled_source_indices,
        }

        return source_agg, target_agg

    def subsample_cifar(self, source_agg, target_agg):
        source_agg["source_indices"] = None
        return source_agg, target_agg

    def subsample_tiny_imagenet(self, source_agg, target_agg):
        source_agg["source_indices"] = None
        return source_agg, target_agg

    def get_classid2indices(self, labels):
        classid2indices = defaultdict(list)
        for i, class_idx in enumerate(labels):
            classid2indices[class_idx.item()].append(i)

        return classid2indices

    def labels_remmaping(self, labels, keep_classes):
        mapping = {
            old_class.item(): new_class
            for new_class, old_class in enumerate(keep_classes)
        }
        remapped_labels = torch.tensor([mapping[label.item()] for label in labels])

        return remapped_labels

    def subsample_iwildcam(self, source_agg, target_agg):
        # Samping based on target distribution
        bincount = torch.bincount(target_agg["y_true"])
        # This gives us top-20 most frequent classes
        keep_classes = torch.argsort(bincount, descending=True)[:20]
        per_class_subsample_size = bincount[keep_classes].min().item()
        # Subsample target
        classid2indices = self.get_classid2indices(target_agg["y_true"])
        # Sample indices
        sampled_target_indices = []
        for class_idx in keep_classes.tolist():
            indices = classid2indices[class_idx]
            per_class_sampled_indices = np.random.choice(
                indices, size=per_class_subsample_size, replace=False
            )
            sampled_target_indices.extend(per_class_sampled_indices)
        # Prepare target
        target_logits = target_agg["y_logits"][sampled_target_indices][:, keep_classes]
        target_labels = self.labels_remmaping(
            target_agg["y_true"][sampled_target_indices], keep_classes=keep_classes
        )
        subsampled_target_agg = {"y_logits": target_logits, "y_true": target_labels}
        # Subsample source
        classid2indices = self.get_classid2indices(source_agg["y_true"])
        # Sample indices
        sampled_source_indices = []
        for class_idx in keep_classes.tolist():
            indices = classid2indices[class_idx]
            sampled_source_indices.extend(indices)
        # Subsample source agg
        source_logits = source_agg["y_logits"][sampled_source_indices][:, keep_classes]
        source_labels = self.labels_remmaping(
            source_agg["y_true"][sampled_source_indices], keep_classes=keep_classes
        )
        subsampled_source_agg = {
            "y_logits": source_logits,
            "y_true": source_labels,
            "source_indices": sampled_source_indices,
        }

        return subsampled_source_agg, subsampled_target_agg

    def subsample_amazon(self, source_agg, target_agg):
        # Get per-class counts & indices
        per_class_counts = torch.bincount(target_agg["y_true"])
        per_class_subsample_size = per_class_counts.min().item()
        # Get per-class counts & indices
        classid2indices = defaultdict(list)
        for i, class_idx in enumerate(target_agg["y_true"]):
            classid2indices[class_idx.item()].append(i)
        # Sample indices
        sampled_indices = []
        for class_idx, indices in classid2indices.items():
            # Filter 0-count classes
            if len(indices) < per_class_subsample_size:
                continue
            per_class_sampled_indices = np.random.choice(
                indices, size=per_class_subsample_size, replace=False
            )
            sampled_indices.extend(per_class_sampled_indices)
        # Subsample target agg
        subsampled_target_agg = {
            "y_true": target_agg["y_true"][sampled_indices],
            "y_logits": target_agg["y_logits"][sampled_indices],
        }
        source_agg["source_indices"] = None

        return source_agg, subsampled_target_agg

    def __call__(self, dataset_name: str, source_agg, target_agg):
        return self.dataset2subsampler[dataset_name](source_agg, target_agg)
