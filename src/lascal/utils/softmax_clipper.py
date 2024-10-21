import numpy as np
import torch


class SoftmaxClipper:

    def __init__(self):
        self.method2value = {
            "em-bcts": 1e-15,
            "elsa": 1e-3,
            "bbse-hard": 1e-2,
            "prob_match": 1e-15,
        }

    def __call__(self, val_softmax_preds, test_softmax_preds, method):
        if isinstance(val_softmax_preds, torch.Tensor):
            val_softmax_preds = val_softmax_preds.numpy()
        if isinstance(test_softmax_preds, torch.Tensor):
            test_softmax_preds = test_softmax_preds.numpy()
        if method not in self.method2value:
            return val_softmax_preds, test_softmax_preds
        # Clip
        val_softmax_preds = np.clip(
            val_softmax_preds, a_min=self.method2value[method], a_max=None
        )
        test_softmax_preds = np.clip(
            test_softmax_preds, a_min=self.method2value[method], a_max=None
        )

        return val_softmax_preds, test_softmax_preds
