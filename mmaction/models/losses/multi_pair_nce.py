import torch

from ..registry import LOSSES
from .base import BaseWeightedLoss


@LOSSES.register_module()
class MultiPairNCE(BaseWeightedLoss):

    def __init__(self, **kwargs):
        super(MultiPairNCE, self).__init__(**kwargs)

    def _forward(self, logits, labels, **kwargs):
        """

        Args:
            logits (torch.Tensor): [NxT, NxT+K]
            labels (torch.Tensor): [NXT, NxT+K]

        Returns:

        """
        row_maxes = torch.max(logits, dim=-1, keepdim=True)[0]
        scaled_logits = logits - row_maxes
        mask = labels.bool()
        labels = labels.to(dtype=scaled_logits.dtype)
        # pos_logits = scaled_logits * labels - 2**20 * (1 - labels)
        # neg_logits = scaled_logits * (1 - labels) - 2**20 * labels
        # log_softmax = pos_logits - torch.log(
        #     pos_logits.exp() + neg_logits.exp().sum(dim=-1, keepdim=True))
        log_softmax = scaled_logits - torch.log(
            scaled_logits.exp() * labels +
            (scaled_logits.exp() * (1 - labels)).sum(dim=-1, keepdim=True))
        loss = -log_softmax[mask]

        return loss
