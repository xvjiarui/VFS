import torch
import torch.nn.functional as F

from ..registry import LOSSES
from .base import BaseWeightedLoss


@LOSSES.register_module()
class DotSimLoss(BaseWeightedLoss):
    """NLL Loss.

    It will calculate Dor Product Similarity loss given cls_score and label.
    """

    def _forward(self, cls_score, label, **kwargs):
        batches, channels, height, width = cls_score.size()
        prod = torch.bmm(
            cls_score.view(batches, channels,
                           height * width).permute(0, 2, 1).contiguous(),
            label.view(batches, channels, height * width))
        loss = -prod.mean()
        return loss


@LOSSES.register_module()
class CosineSimLoss(BaseWeightedLoss):
    """NLL Loss.

    It will calculate Cosine Similarity loss given cls_score and label.
    """

    def __init__(self,
                 with_norm=True,
                 negative=False,
                 pairwise=False,
                 **kwargs):
        super().__init__(**kwargs)
        self.with_norm = with_norm
        self.negative = negative
        self.pairwise = pairwise

    def _forward(self, cls_score, label, mask=None, **kwargs):
        if self.with_norm:
            cls_score = F.normalize(cls_score, p=2, dim=1)
            label = F.normalize(label, p=2, dim=1)
        if mask is not None:
            assert self.pairwise
        if self.pairwise:
            cls_score = cls_score.flatten(2)
            label = label.flatten(2)
            prod = torch.einsum('bci,bcj->bij', cls_score, label)
            if mask is not None:
                assert prod.shape == mask.shape
                prod *= mask.float()
            prod = prod.flatten(1)
        else:
            prod = torch.sum(
                cls_score * label, dim=1).view(cls_score.size(0), -1)
        if self.negative:
            loss = -prod.mean(dim=-1)
        else:
            loss = 2 - 2 * prod.mean(dim=-1)
        return loss
