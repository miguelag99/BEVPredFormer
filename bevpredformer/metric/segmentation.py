from typing import Optional

import torch
from einops import rearrange
from torchmetrics.metric import Metric


class IoUMetric(Metric):
    def __init__(
        self,
        thresholds: float = 0.5,
        min_value_mask: Optional[int] = None,
        exact_value_mask: Optional[int] = None,
    ):
        """
        label_indices:
            transforms labels (c, h, w) to (len(labels), h, w)
            see config/experiment/* for examples
        min_value_mask:
            passing "None" will ignore the mask
            otherwise uses visibility values to ignore certain labels
        """
        super().__init__(dist_sync_on_step=False, compute_on_step=False)

        self.thresholds = thresholds
        self.exact_value_mask = exact_value_mask
        self.min_value_mask = min_value_mask

        self.add_state("tp", default=torch.zeros(1), dist_reduce_fx="sum")
        self.add_state("fp", default=torch.zeros(1), dist_reduce_fx="sum")
        self.add_state("fn", default=torch.zeros(1), dist_reduce_fx="sum")

    def _get_mask(self, mask, shape, device):
        if self.exact_value_mask is not None:
            assert mask is not None
            mask = mask == self.exact_value_mask

        elif self.min_value_mask is not None:
            assert mask is not None
            mask = mask >= self.min_value_mask

        elif mask is not None:
            mask = mask

        else:
            mask = torch.ones(shape, device=device, dtype=torch.float32)
        return mask

    def update(self, pred, label, mask=None):
        """Mask: 1 to keep, 0 to discard."""
        assert pred.shape[1] in [1, 2], f"Expected 1 or 2 channels, got {pred.shape[1]}"
        # Handle multiclass for vehicle and background ONLY!!!
        if pred.shape[1] == 2:
            pred = torch.softmax(pred,dim=1)[:,1].contiguous()
        mask = self._get_mask(mask, pred.shape, pred.device)
        pred = pred.detach().view(-1, 1)
        label = label.detach().bool().view(-1, 1)
        mask = mask.detach().float().view(-1, 1)

        pred = pred >= self.thresholds

        self.tp += ((pred & label) * mask).sum(0)
        self.fp += ((pred & ~label) * mask).sum(0)
        self.fn += ((~pred & label) * mask).sum(0)

    def compute(self, eps=1e-7):
        return self.tp / (self.tp + self.fp + self.fn + eps)



from torchmetrics.classification import MulticlassStatScores


class IntersectionOverUnion(Metric):
    """Computes intersection-over-union."""
    def __init__(
        self,
        n_classes: int,
        ignore_index: Optional[int] = None,
        absent_score: float = 0.0,
        reduction: str = 'none',
    ):
        super().__init__()

        self.n_classes = n_classes
        self.ignore_index = ignore_index
        self.absent_score = absent_score
        self.reduction = reduction

        self.mcss = MulticlassStatScores(num_classes=n_classes,
                                         ignore_index=ignore_index,
                                         multidim_average='global',
                                         average=None)


        self.add_state('true_positive', default=torch.zeros(n_classes), dist_reduce_fx='sum')
        self.add_state('false_positive', default=torch.zeros(n_classes), dist_reduce_fx='sum')
        self.add_state('false_negative', default=torch.zeros(n_classes), dist_reduce_fx='sum')
        self.add_state('support', default=torch.zeros(n_classes), dist_reduce_fx='sum')

    def update(self, prediction: torch.Tensor, target: torch.Tensor):
        
        stats = self.mcss(prediction, target)
        tps = stats[:,0]
        fps = stats[:,1]
        fns = stats[:,3]
        sups = stats[:,4]
        # tps, fps, _, fns, sups = self.mcss(prediction, target)

        self.true_positive += tps
        self.false_positive += fps
        self.false_negative += fns
        self.support += sups

    def compute(self):
        scores = torch.zeros(self.n_classes, device=self.true_positive.device, dtype=torch.float32)

        for class_idx in range(self.n_classes):
            if class_idx == self.ignore_index:
                continue

            tp = self.true_positive[class_idx]
            fp = self.false_positive[class_idx]
            fn = self.false_negative[class_idx]
            sup = self.support[class_idx]

            # If this class is absent in the target (no support) AND absent in the pred (no true or false
            # positives), then use the absent_score for this class.
            if sup + tp + fp == 0:
                scores[class_idx] = self.absent_score
                continue

            denominator = tp + fp + fn
            score = tp.to(torch.float) / denominator
            scores[class_idx] = score

        # Remove the ignored class index from the scores.
        if (self.ignore_index is not None) and (0 <= self.ignore_index < self.n_classes):
            scores = torch.cat([scores[:self.ignore_index], scores[self.ignore_index+1:]])

        return scores