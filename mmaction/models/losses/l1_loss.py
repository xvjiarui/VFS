import torch.nn.functional as F

from ..registry import LOSSES
from .base import BaseWeightedLoss


@LOSSES.register_module()
class L1Loss(BaseWeightedLoss):
    """NLL Loss.

    It will calculate NLL loss given cls_score and label.
    """

    def _forward(self, cls_score, label, **kwargs):
        """Forward function.

        Args:
            cls_score (torch.Tensor): The class score.
            label (torch.Tensor): The ground truth label.
            kwargs: Any keyword argument to be used to calculate nll loss.

        Returns:
            torch.Tensor: The returned nll loss.
        """
        loss_cls = F.l1_loss(cls_score, label, **kwargs)
        return loss_cls


@LOSSES.register_module()
class SmoothL1Loss(BaseWeightedLoss):
    """NLL Loss.

    It will calculate NLL loss given cls_score and label.
    """

    def _forward(self, cls_score, label, **kwargs):
        """Forward function.

        Args:
            cls_score (torch.Tensor): The class score.
            label (torch.Tensor): The ground truth label.
            kwargs: Any keyword argument to be used to calculate nll loss.

        Returns:
            torch.Tensor: The returned nll loss.
        """
        loss_cls = F.smooth_l1_loss(cls_score, label, **kwargs)
        return loss_cls
