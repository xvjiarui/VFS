import torch.nn.functional as F
from mmcv.ops.point_sample import generate_grid
from torch.nn.modules.utils import _pair

from ..common import compute_affinity, propagate
from ..registry import LOSSES
from .base import BaseWeightedLoss


def im2col(img, win_len, stride=1):
    """
    Args:
        img: a b*c*h*w feature tensor.
        win_len: each pixel compares with its neighbors within
            a (win_len*2+1) * (win_len*2+1) window.
        stride: stride of unfold

    Returns:
        result: a b*c*(h*w)*(win_len*2+1)^2 tensor, unfolded neighbors for
            each pixel
    """
    b, c, _, _ = img.size()
    # b * (c*w*w) * win_num
    stride = _pair(stride)
    unfold_img = F.unfold(img, win_len, stride=stride)
    unfold_img = unfold_img.view(b, c, win_len * win_len, -1)
    unfold_img = unfold_img.permute(0, 1, 3, 2).contiguous()
    return unfold_img


@LOSSES.register_module()
class ConcentrateLoss(BaseWeightedLoss):
    """NLL Loss.

    It will calculate Cosine Similarity loss given cls_score and label.
    """

    def __init__(self,
                 win_len=8,
                 stride=8,
                 temperature=1.,
                 with_norm=True,
                 **kwargs):
        super().__init__(**kwargs)
        self.win_len = win_len
        self.stride = stride
        self.temperature = temperature
        self.with_norm = with_norm

    def _forward(self, src_x, dst_x, **kwargs):
        assert src_x.shape == dst_x.shape, f'{src_x.shape} vs {dst_x.shape}'
        batches, channels, height, width = src_x.size()
        affinity = compute_affinity(
            src_x,
            dst_x,
            normalize=self.with_norm,
            temperature=self.temperature)

        grid = generate_grid(
            src_x.size(0), src_x.shape[2:], device=src_x.device)
        # [N, 2, H, W]
        grid = grid.permute(0, 2, 1).reshape(batches, 2, height, width)

        # [N, 2, H, W]
        grid_dst = propagate(grid, affinity.softmax(1))
        grid_src = propagate(grid,
                             affinity.permute(0, 2, 1).contiguous().softmax(1))

        # [N, 2, H*W, win^2]
        grid_unfold_dst = im2col(grid_dst, self.win_len, self.stride)
        grid_unfold_src = im2col(grid_src, self.win_len, self.stride)

        # [N, 2, H*W, 1]
        center_dst = grid_unfold_dst.mean(dim=3, keepdims=True)
        dist_dst = (grid_unfold_dst - center_dst)**2
        center_src = grid_unfold_src.mean(dim=3, keepdims=True)
        dist_src = (grid_unfold_src - center_src)**2

        loss = dist_dst.mean() + dist_src.mean()

        return loss


@LOSSES.register_module()
class AffinityConcentrateLoss(BaseWeightedLoss):
    """NLL Loss.

    It will calculate Cosine Similarity loss given cls_score and label.
    """

    def __init__(self, win_len=8, stride=8, **kwargs):
        super().__init__(**kwargs)
        self.win_len = win_len
        self.stride = stride

    def _forward(self, affinity, **kwargs):
        # suppose affinity is square
        batches = affinity.size(0)
        height = int(affinity.size(1)**0.5)
        width = height
        grid = generate_grid(batches, (height, width), device=affinity.device)
        # [N, 2, H, W]
        grid = grid.permute(0, 2, 1).reshape(batches, 2, height, width)

        # [N, 2, H, W]
        grid_dst = propagate(grid, affinity.softmax(1))
        grid_src = propagate(grid,
                             affinity.permute(0, 2, 1).contiguous().softmax(1))

        # [N, 2, H*W, win^2]
        grid_unfold_dst = im2col(grid_dst, self.win_len, self.stride)
        grid_unfold_src = im2col(grid_src, self.win_len, self.stride)

        # [N, 2, H*W, 1]
        center_dst = grid_unfold_dst.mean(dim=3, keepdims=True)
        dist_dst = (grid_unfold_dst - center_dst)**2
        center_src = grid_unfold_src.mean(dim=3, keepdims=True)
        dist_src = (grid_unfold_src - center_src)**2

        loss = dist_dst.mean() + dist_src.mean()

        return loss
