from typing import List

import mmcv
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair, _single, _triple


def change_stride(conv, stride):
    """Inplace change conv stride.

    Args:
        conv (nn.Module):
        stride (int):
    """
    if isinstance(conv, nn.Conv1d):
        conv.stride = _single(stride)
    if isinstance(conv, nn.Conv2d):
        conv.stride = _pair(stride)
    if isinstance(conv, nn.Conv3d):
        conv.stride = _triple(stride)


def pil_nearest_interpolate(input, size):
    # workaround for https://github.com/pytorch/pytorch/issues/34808
    resized_imgs = []
    input = input.permute(0, 2, 3, 1)
    for img in input:
        img = img.squeeze(-1)
        img = img.detach().cpu().numpy()
        resized_img = mmcv.imresize(
            img,
            size=(size[1], size[0]),
            interpolation='nearest',
            backend='pillow')
        resized_img = torch.from_numpy(resized_img).to(
            input, non_blocking=True)
        resized_img = resized_img.unsqueeze(2).permute(2, 0, 1)
        resized_imgs.append(resized_img)

    return torch.stack(resized_imgs, dim=0)


def video2images(imgs):
    batches, channels, clip_len = imgs.shape[:3]
    if clip_len == 1:
        new_imgs = imgs.squeeze(2).reshape(batches, channels, *imgs.shape[3:])
    else:
        new_imgs = imgs.transpose(1, 2).contiguous().reshape(
            batches * clip_len, channels, *imgs.shape[3:])

    return new_imgs


def images2video(imgs, clip_len):
    batches, channels = imgs.shape[:2]
    if clip_len == 1:
        new_imgs = imgs.unsqueeze(2)
    else:
        new_imgs = imgs.reshape(batches // clip_len, clip_len, channels,
                                *imgs.shape[2:]).transpose(1, 2).contiguous()

    return new_imgs


# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """Performs all_gather operation on the provided tensors.

    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [
        torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())
    ]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output


class StrideContext(object):

    def __init__(self, backbone, strides, out_indices=None):
        self.backbone = backbone
        self.strides = strides
        self.out_indices = out_indices

    def __enter__(self):
        if self.strides is not None:
            self.backbone.switch_strides(self.strides)
        if self.out_indices is not None:
            self.backbone.switch_out_indices(self.out_indices)

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.strides is not None:
            self.backbone.switch_strides()
        if self.out_indices is not None:
            self.backbone.switch_out_indices()


def unmap(data, count, inds, fill=0):
    """Unmap a subset of item (data) back to the original set of items (of size
    count)"""
    if data.dim() == 1:
        ret = data.new_full((count, ), fill)
        ret[inds.type(torch.bool)] = data
    else:
        new_size = (count, ) + data.size()[1:]
        ret = data.new_full(new_size, fill)
        ret[inds.type(torch.bool), :] = data
    return ret


@torch.no_grad()
def _batch_shuffle_ddp(x):
    """Batch shuffle, for making use of BatchNorm.

    *** Only support DistributedDataParallel (DDP) model. ***
    """
    # gather from all gpus
    batch_size_this = x.shape[0]
    x_gather = concat_all_gather(x)
    batch_size_all = x_gather.shape[0]

    num_gpus = batch_size_all // batch_size_this

    # random shuffle index
    idx_shuffle = torch.randperm(batch_size_all).cuda()

    # broadcast to all gpus
    torch.distributed.broadcast(idx_shuffle, src=0)

    # index for restoring
    idx_unshuffle = torch.argsort(idx_shuffle)

    # shuffled index for this gpu
    gpu_idx = torch.distributed.get_rank()
    idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

    return x_gather[idx_this], idx_unshuffle


@torch.no_grad()
def _batch_unshuffle_ddp(x, idx_unshuffle):
    """Undo batch shuffle.

    *** Only support DistributedDataParallel (DDP) model. ***
    """
    # gather from all gpus
    batch_size_this = x.shape[0]
    x_gather = concat_all_gather(x)
    batch_size_all = x_gather.shape[0]

    num_gpus = batch_size_all // batch_size_this

    # restored index for this gpu
    gpu_idx = torch.distributed.get_rank()
    idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

    return x_gather[idx_this]


class Clamp(nn.Module):

    def __init__(self, min=None, max=None):
        super(Clamp, self).__init__()
        self.min = min
        self.max = max
        assert self.min is not None or self.max is not None

    def forward(self, x):
        kwargs = {}
        if self.min is not None:
            kwargs['min'] = self.min
        if self.max is not None:
            kwargs['max'] = self.max
        return x.clamp(**kwargs)

    def extra_repr(self):
        """Extra repr."""
        s = f'min={self.min}, max={self.max}'
        return s


def cat(tensors: List[torch.Tensor], dim: int = 0):
    """Efficient version of torch.cat that avoids a copy if there is only a
    single element in a list."""
    assert isinstance(tensors, (list, tuple))
    if len(tensors) == 1:
        return tensors[0]
    return torch.cat(tensors, dim)


def normalize_logit(seg_logit):
    seg_logit_min = seg_logit.view(*seg_logit.shape[:2], -1).min(
        dim=-1)[0].view(*seg_logit.shape[:2], 1, 1)
    seg_logit_max = seg_logit.view(*seg_logit.shape[:2], -1).max(
        dim=-1)[0].view(*seg_logit.shape[:2], 1, 1)
    normalized_seg_logit = (seg_logit - seg_logit_min) / (
        seg_logit_max - seg_logit_min + 1e-12)
    seg_logit = torch.where(seg_logit_max > 0, normalized_seg_logit, seg_logit)

    return seg_logit


def mean_list(input_list):
    ret = input_list[0].clone()
    for i in range(1, len(input_list)):
        ret += input_list[i]
    ret /= len(input_list)
    return ret


def interpolate3d(input,
                  size=None,
                  scale_factor=None,
                  mode='nearest',
                  align_corners=False):
    results = []
    clip_len = input.size(2)
    for i in range(clip_len):
        results.append(
            F.interpolate(
                input[:, :, i],
                size=size,
                scale_factor=scale_factor,
                mode=mode,
                align_corners=align_corners))

    return torch.stack(results, dim=2)
