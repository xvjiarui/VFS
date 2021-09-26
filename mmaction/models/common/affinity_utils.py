import torch
import torch.nn.functional as F
from torch.nn.modules.utils import _pair


def compute_affinity(src_img,
                     dst_img,
                     temperature=1.,
                     normalize=True,
                     softmax_dim=None,
                     mask=None):
    batches, channels = src_img.shape[:2]
    src_feat = src_img.view(batches, channels, src_img.shape[2:].numel())
    dst_feat = dst_img.view(batches, channels, dst_img.shape[2:].numel())
    if normalize:
        src_feat = F.normalize(src_feat, p=2, dim=1)
        dst_feat = F.normalize(dst_feat, p=2, dim=1)
    src_feat = src_feat.permute(0, 2, 1).contiguous()
    dst_feat = dst_feat.contiguous()
    affinity = torch.bmm(src_feat, dst_feat) / temperature
    if mask is not None:
        # affinity -= (1 - mask) * 2**30
        affinity.masked_fill_(~mask.bool(), float('-inf'))
    if softmax_dim is not None:
        affinity = affinity.softmax(dim=softmax_dim)

    if mask is not None:
        affinity[affinity.isnan()] = 0

    return affinity


def propagate(img, affinity, topk=None):
    batches, channels, height, width = img.size()
    if topk is not None:
        # tk_val, tk_idx = affinity.topk(dim=1, k=topk)
        # tk_val_min, _ = tk_val.min(dim=1)
        tk_val_min = affinity.topk(dim=1, k=topk)[0][:, topk - 1]
        tk_val_min = tk_val_min.view(batches, 1, height * width)
        # use in-place ops to save memory
        affinity -= tk_val_min
        affinity.clamp_(min=0)
        # TODO check
        affinity /= affinity.sum(keepdim=True, dim=1).clamp(min=1e-12)
    img = img.view(batches, channels, -1)
    img = img.contiguous()
    affinity = affinity.contiguous()
    new_img = torch.bmm(img, affinity).contiguous()
    new_img = new_img.reshape(batches, channels, height, width)
    return new_img


def propagate_temporal(imgs, affinities, topk=None):
    batches, channels, clip_len, height, width = imgs.size()
    assert affinities.size(0) == batches
    assert affinities.size(1) == clip_len
    assert affinities.size(2) == height * width
    assert affinities.size(2) == affinities.size(3)
    affinities = affinities.reshape(batches, clip_len * height * width,
                                    height * width)
    if topk is not None:
        # tk_val, _ = affinities.topk(dim=1, k=topk)
        # tk_val_min, _ = tk_val.min(dim=1)
        tk_val_min = affinities.topk(dim=1, k=topk)[0][:, topk - 1]
        tk_val_min = tk_val_min.view(batches, 1, height * width)
        # use in-place ops to save memory
        affinities -= tk_val_min
        affinities.clamp_(min=0)
        affinities /= affinities.sum(keepdim=True, dim=1).clamp(min=1e-12)
    imgs = imgs.reshape(batches, channels, -1)
    new_imgs = torch.bmm(imgs, affinities)
    new_imgs = new_imgs.reshape(batches, channels, height, width)
    return new_imgs


#
#
# def propagate_temporal_naive(imgs, affinities, topk=None):
#     batches, channels, clip_len, height, width = imgs.size()
#     assert affinities.size(0) == batches
#     assert affinities.size(1) == clip_len
#     assert affinities.size(2) == height * width
#     assert affinities.size(2) == affinities.size(3)
#     affinities = affinities.reshape(batches, clip_len * height * width,
#                                     height * width)
#     new_imgs = imgs.new_zeros(batches, channels, height*width)
#     num_chunks = 4 * clip_len
#     chunk_size = height * width // num_chunks
#     for i in range(num_chunks):
#         new_imgs[:, :, i * chunk_size:(i + 1) * chunk_size] =
#         _propagate_chunk(
#             imgs, affinities[:, :, i * chunk_size:(i + 1) * chunk_size],
#             topk=topk)
#     # handle remaining
#     if height * width % chunk_size != 0:
#         new_imgs[:, :, num_chunks * chunk_size:] = _propagate_chunk(
#             imgs, affinities[:, :, num_chunks*chunk_size:],
#             topk=topk)
#     new_imgs = new_imgs.reshape(batches, channels, height, width)
#
#     return new_imgs

# def _propagate_chunk(imgs, affinities, topk=None):
#     batches, channels = imgs.shape[:2]
#     if topk is not None:
#         affinities = affinities.clone()
#         tk_val, tk_idx = affinities.topk(dim=1, k=topk)
#         tk_val_min, _ = tk_val.min(dim=1)
#         tk_val_min = tk_val_min.view(batches, 1, -1)
#         affinities[tk_val_min > affinities] = 0
#     imgs = imgs.reshape(batches, channels, -1)
#     assert imgs.size(2) == affinities.size(1)
#     new_img = torch.bmm(imgs, affinities)
#
#     return new_img
#


def spatial_neighbor(batches,
                     height,
                     width,
                     neighbor_range,
                     device,
                     dtype,
                     dim=1,
                     mode='circle'):
    assert dim in [1, 2]
    assert mode in ['circle', 'square']
    if mode == 'square':
        neighbor_range = _pair(neighbor_range)
        mask = torch.zeros(
            batches, height, width, height, width, device=device, dtype=dtype)
        for i in range(height):
            for j in range(width):
                top = max(0, i - neighbor_range[0] // 2)
                left = max(0, j - neighbor_range[1] // 2)
                bottom = min(height, i + neighbor_range[0] // 2 + 1)
                right = min(width, j + neighbor_range[1] // 2 + 1)
                mask[:, top:bottom, left:right, i, j] = 1

        mask = mask.view(batches, height * width, height * width)
        if dim == 2:
            mask = mask.transpose(1, 2).contiguous()
    else:
        radius = neighbor_range // 2
        grid_x, grid_y = torch.meshgrid(
            torch.arange(height, device=device, dtype=dtype),
            torch.arange(width, device=device, dtype=dtype))
        dist_mat = ((grid_x.view(height, width, 1, 1) -
                     grid_x.view(1, 1, height, width))**2 +
                    (grid_y.view(height, width, 1, 1) -
                     grid_y.view(1, 1, height, width))**2)**0.5
        mask = dist_mat < radius
        mask = mask.view(height * width, height * width)
        mask = mask.to(device=device, dtype=dtype)
    return mask.bool()


def resize_spatial_mask(mask, output_size):
    height, width = mask.shape[:2]
    mask = mask.view(1, height * width, height, width).byte()
    new_mask = F.interpolate(mask, size=output_size)
    new_mask = new_mask.view(height, width, *output_size)
    return new_mask


def grid_mask(grid1, grid2, radius, diag_norm=224):
    dist = torch.pow(
        grid1.view(*grid1.shape[:2], -1, 1) -
        grid2.view(*grid2.shape[:2], 1, -1), 2).sum(dim=1)**0.5
    dist *= (grid2.size(2)**2 + grid2.size(3)**2)**0.5 / (diag_norm * 2**0.5)
    mask = dist < radius

    return mask
