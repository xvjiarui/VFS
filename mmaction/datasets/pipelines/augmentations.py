import random
from collections.abc import Sequence

import mmcv
import numpy as np
from numpy import random as npr
from PIL import Image, ImageFilter
from skimage.util import view_as_windows
from torch.nn.modules.utils import _pair
from torchvision.transforms import ColorJitter as _ColorJitter
from torchvision.transforms import RandomAffine as _RandomAffine
from torchvision.transforms import RandomResizedCrop as _RandomResizedCrop
from torchvision.transforms import functional as F

from ..registry import PIPELINES


def _init_lazy_if_proper(results, lazy):
    """Initialize lazy operation properly.

    Make sure that a lazy operation is properly initialized,
    and avoid a non-lazy operation accidentally getting mixed in.

    Required keys in results are "imgs" if "img_shape" not in results,
    otherwise, Required keys in results are "img_shape", add or modified keys
    are "img_shape", "lazy".
    Add or modified keys in "lazy" are "original_shape", "crop_bbox", "flip",
    "flip_direction", "interpolation".

    Args:
        results (dict): A dict stores data pipeline result.
        lazy (bool): Determine whether to apply lazy operation. Default: False.
    """

    if 'img_shape' not in results:
        results['img_shape'] = results['imgs'][0].shape[:2]
    if lazy:
        if 'lazy' not in results:
            img_h, img_w = results['img_shape']
            lazyop = dict()
            lazyop['original_shape'] = results['img_shape']
            lazyop['crop_bbox'] = np.array([0, 0, img_w, img_h],
                                           dtype=np.float32)
            lazyop['flip'] = False
            lazyop['flip_direction'] = None
            lazyop['interpolation'] = None
            results['lazy'] = lazyop
    else:
        assert 'lazy' not in results, 'Use Fuse after lazy operations'


@PIPELINES.register_module()
class Fuse(object):
    """Fuse lazy operations.

    Fusion order:
        crop -> resize -> flip

    Required keys are "imgs", "img_shape" and "lazy", added or modified keys
    are "imgs", "lazy".
    Required keys in "lazy" are "crop_bbox", "interpolation", "flip_direction".
    """

    def __call__(self, results):
        if 'lazy' not in results:
            raise ValueError('No lazy operation detected')
        lazyop = results['lazy']
        imgs = results['imgs']

        # crop
        left, top, right, bottom = lazyop['crop_bbox'].round().astype(int)
        imgs = [img[top:bottom, left:right] for img in imgs]

        # resize
        img_h, img_w = results['img_shape']
        if lazyop['interpolation'] is None:
            interpolation = 'bilinear'
        else:
            interpolation = lazyop['interpolation']
        imgs = [
            mmcv.imresize(img, (img_w, img_h), interpolation=interpolation)
            for img in imgs
        ]

        # flip
        if lazyop['flip']:
            for img in imgs:
                mmcv.imflip_(img, lazyop['flip_direction'])

        results['imgs'] = imgs
        del results['lazy']

        return results


@PIPELINES.register_module()
class RandomCrop(object):
    """Vanilla square random crop that specifics the output size.

    Required keys in results are "imgs" and "img_shape", added or
    modified keys are "imgs", "lazy"; Required keys in "lazy" are "flip",
    "crop_bbox", added or modified key is "crop_bbox".

    Args:
        size (int): The output size of the images.
        lazy (bool): Determine whether to apply lazy operation. Default: False.
    """

    def __init__(self, size, lazy=False):
        if not isinstance(size, int):
            raise TypeError(f'Size must be an int, but got {type(size)}')
        self.size = size
        self.lazy = lazy

    def __call__(self, results):
        """Performs the RandomCrop augmentation.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        _init_lazy_if_proper(results, self.lazy)

        img_h, img_w = results['img_shape']
        assert self.size <= img_h and self.size <= img_w

        y_offset = 0
        x_offset = 0
        if img_h > self.size:
            y_offset = int(np.random.randint(0, img_h - self.size))
        if img_w > self.size:
            x_offset = int(np.random.randint(0, img_w - self.size))

        new_h, new_w = self.size, self.size

        results['crop_bbox'] = np.array(
            [x_offset, y_offset, x_offset + new_w, y_offset + new_h])
        results['img_shape'] = (new_h, new_w)

        if not self.lazy:
            results['imgs'] = [
                img[y_offset:y_offset + new_h, x_offset:x_offset + new_w]
                for img in results['imgs']
            ]
        else:
            lazyop = results['lazy']
            if lazyop['flip']:
                raise NotImplementedError('Put Flip at last for now')

            # record crop_bbox in lazyop dict to ensure only crop once in Fuse
            lazy_left, lazy_top, lazy_right, lazy_bottom = lazyop['crop_bbox']
            left = x_offset * (lazy_right - lazy_left) / img_w
            right = (x_offset + new_w) * (lazy_right - lazy_left) / img_w
            top = y_offset * (lazy_bottom - lazy_top) / img_h
            bottom = (y_offset + new_h) * (lazy_bottom - lazy_top) / img_h
            lazyop['crop_bbox'] = np.array([(lazy_left + left),
                                            (lazy_top + top),
                                            (lazy_left + right),
                                            (lazy_top + bottom)],
                                           dtype=np.float32)

        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}(size={self.size}, '
                    f'lazy={self.lazy})')
        return repr_str


@PIPELINES.register_module()
class RandomResizedCrop(object):
    """Random crop that specifics the area and height-weight ratio range.

    Required keys in results are "imgs", "img_shape", "crop_bbox" and "lazy",
    added or modified keys are "imgs", "crop_bbox" and "lazy"; Required keys
    in "lazy" are "flip", "crop_bbox", added or modified key is "crop_bbox".

    Args:
        area_range (Tuple[float]): The candidate area scales range of
            output cropped images. Default: (0.08, 1.0).
        aspect_ratio_range (Tuple[float]): The candidate aspect ratio range of
            output cropped images. Default: (3 / 4, 4 / 3).
        lazy (bool): Determine whether to apply lazy operation. Default: False.
    """

    def __init__(self,
                 area_range=(0.08, 1.0),
                 aspect_ratio_range=(3 / 4, 4 / 3),
                 same_on_clip=True,
                 same_across_clip=True,
                 same_clip_indices=None,
                 same_frame_indices=None,
                 lazy=False):
        self.area_range = area_range
        self.aspect_ratio_range = aspect_ratio_range
        self.lazy = lazy
        if not mmcv.is_tuple_of(self.area_range, float):
            raise TypeError(f'Area_range must be a tuple of float, '
                            f'but got {type(area_range)}')
        if not mmcv.is_tuple_of(self.aspect_ratio_range, float):
            raise TypeError(f'Aspect_ratio_range must be a tuple of float, '
                            f'but got {type(aspect_ratio_range)}')
        self.same_on_clip = same_on_clip
        self.same_across_clip = same_across_clip
        if same_clip_indices is not None:
            assert isinstance(same_clip_indices, Sequence)
        self.same_clip_indices = same_clip_indices
        if same_frame_indices is not None:
            assert isinstance(same_frame_indices, Sequence)
        self.same_frame_indices = same_frame_indices

    @staticmethod
    def get_crop_bbox(img_shape,
                      area_range,
                      aspect_ratio_range,
                      max_attempts=10):
        """Get a crop bbox given the area range and aspect ratio range.

        Args:
            img_shape (Tuple[int]): Image shape
            area_range (Tuple[float]): The candidate area scales range of
                output cropped images. Default: (0.08, 1.0).
            aspect_ratio_range (Tuple[float]): The candidate aspect
                ratio range of output cropped images. Default: (3 / 4, 4 / 3).
                max_attempts (int): The maximum of attempts. Default: 10.
            max_attempts (int): Max attempts times to generate random candidate
                bounding box. If it doesn't qualified one, the center bounding
                box will be used.
        Returns:
            (list[int]) A random crop bbox within the area range and aspect
            ratio range.
        """
        assert 0 < area_range[0] <= area_range[1] <= 1
        assert 0 < aspect_ratio_range[0] <= aspect_ratio_range[1]

        img_h, img_w = img_shape
        area = img_h * img_w

        min_ar, max_ar = aspect_ratio_range
        aspect_ratios = np.exp(
            np.random.uniform(
                np.log(min_ar), np.log(max_ar), size=max_attempts))
        target_areas = np.random.uniform(*area_range, size=max_attempts) * area
        candidate_crop_w = np.round(np.sqrt(target_areas *
                                            aspect_ratios)).astype(np.int32)
        candidate_crop_h = np.round(np.sqrt(target_areas /
                                            aspect_ratios)).astype(np.int32)

        for i in range(max_attempts):
            crop_w = candidate_crop_w[i]
            crop_h = candidate_crop_h[i]
            if crop_h <= img_h and crop_w <= img_w:
                x_offset = random.randint(0, img_w - crop_w)
                y_offset = random.randint(0, img_h - crop_h)
                return x_offset, y_offset, x_offset + crop_w, y_offset + crop_h

        # Fallback
        crop_size = min(img_h, img_w)
        x_offset = (img_w - crop_size) // 2
        y_offset = (img_h - crop_size) // 2
        return x_offset, y_offset, x_offset + crop_size, y_offset + crop_size

    def __call__(self, results):
        """Performs the RandomResizeCrop augmentation.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        _init_lazy_if_proper(results, self.lazy)

        img_h, img_w = results['img_shape']

        left, top, right, bottom = self.get_crop_bbox(
            (img_h, img_w), self.area_range, self.aspect_ratio_range)
        new_h, new_w = bottom - top, right - left

        results['crop_bbox'] = np.array([left, top, right, bottom])
        results['img_shape'] = (new_h, new_w)

        if not self.lazy:
            for i, img in enumerate(results['imgs']):
                is_new_clip = not self.same_across_clip and i % results[
                    'clip_len'] == 0 and i > 0
                generate_new = not self.same_on_clip or is_new_clip
                if self.same_clip_indices is not None:
                    assert min(self.same_clip_indices) >= 0
                    assert max(self.same_clip_indices) < results['num_clips']
                    keep_same = i // results[
                        'clip_len'] in self.same_clip_indices
                    generate_new = generate_new and not keep_same
                if self.same_frame_indices is not None:
                    assert min(self.same_frame_indices) >= 0
                    assert max(self.same_frame_indices) < results['clip_len']
                    keep_same = i % results[
                        'clip_len'] in self.same_frame_indices
                    generate_new = generate_new and not keep_same
                if generate_new:
                    left, top, right, bottom = self.get_crop_bbox(
                        (img_h, img_w), self.area_range,
                        self.aspect_ratio_range)
                    new_h, new_w = bottom - top, right - left

                results['crop_bbox'] = np.array([left, top, right, bottom])
                results['img_shape'] = (new_h, new_w)
                results['imgs'][i] = img[top:bottom, left:right]
                if 'grids' in results:
                    grid = results['grids'][i]
                    results['grids'][i] = grid[top:bottom, left:right]
        else:
            lazyop = results['lazy']
            if lazyop['flip']:
                raise NotImplementedError('Put Flip at last for now')

            # record crop_bbox in lazyop dict to ensure only crop once in Fuse
            lazy_left, lazy_top, lazy_right, lazy_bottom = lazyop['crop_bbox']
            left = left * (lazy_right - lazy_left) / img_w
            right = right * (lazy_right - lazy_left) / img_w
            top = top * (lazy_bottom - lazy_top) / img_h
            bottom = bottom * (lazy_bottom - lazy_top) / img_h
            lazyop['crop_bbox'] = np.array([(lazy_left + left),
                                            (lazy_top + top),
                                            (lazy_left + right),
                                            (lazy_top + bottom)],
                                           dtype=np.float32)

        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'area_range={self.area_range}, '
                    f'aspect_ratio_range={self.aspect_ratio_range}, '
                    f'lazy={self.lazy})')
        return repr_str


@PIPELINES.register_module()
class MultiScaleCrop(object):
    """Crop images with a list of randomly selected scales.

    Randomly select the w and h scales from a list of scales. Scale of 1 means
    the base size, which is the minimal of image weight and height. The scale
    level of w and h is controlled to be smaller than a certain value to
    prevent too large or small aspect ratio.
    Required keys are "imgs", "img_shape", added or modified keys are "imgs",
    "crop_bbox", "img_shape", "lazy" and "scales". Required keys in "lazy" are
    "crop_bbox", added or modified key is "crop_bbox".

    Args:
        input_size (int | tuple[int]): (w, h) of network input.
        scales (tuple[float]): Weight and height scales to be selected.
        max_wh_scale_gap (int): Maximum gap of w and h scale levels.
            Default: 1.
        random_crop (bool): If set to True, the cropping bbox will be randomly
            sampled, otherwise it will be sampler from fixed regions.
            Default: False.
        num_fixed_crops (int): If set to 5, the cropping bbox will keep 5
            basic fixed regions: "upper left", "upper right", "lower left",
            "lower right", "center".If set to 13, the cropping bbox will append
            another 8 fix regions: "center left", "center right",
            "lower center", "upper center", "upper left quarter",
            "upper right quarter", "lower left quarter", "lower right quarter".
            Default: 5.
        lazy (bool): Determine whether to apply lazy operation. Default: False.
    """

    def __init__(self,
                 input_size,
                 scales=(1, ),
                 max_wh_scale_gap=1,
                 random_crop=False,
                 num_fixed_crops=5,
                 lazy=False):
        self.input_size = _pair(input_size)
        if not mmcv.is_tuple_of(self.input_size, int):
            raise TypeError(f'Input_size must be int or tuple of int, '
                            f'but got {type(input_size)}')

        if not isinstance(scales, tuple):
            raise TypeError(f'Scales must be tuple, but got {type(scales)}')

        if num_fixed_crops not in [5, 13]:
            raise ValueError(f'Num_fix_crops must be in {[5, 13]}, '
                             f'but got {num_fixed_crops}')

        self.scales = scales
        self.max_wh_scale_gap = max_wh_scale_gap
        self.random_crop = random_crop
        self.num_fixed_crops = num_fixed_crops
        self.lazy = lazy

    def __call__(self, results):
        """Performs the MultiScaleCrop augmentation.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        _init_lazy_if_proper(results, self.lazy)

        img_h, img_w = results['img_shape']
        base_size = min(img_h, img_w)
        crop_sizes = [int(base_size * s) for s in self.scales]

        candidate_sizes = []
        for i, h in enumerate(crop_sizes):
            for j, w in enumerate(crop_sizes):
                if abs(i - j) <= self.max_wh_scale_gap:
                    candidate_sizes.append([w, h])

        crop_size = random.choice(candidate_sizes)
        for i in range(2):
            if abs(crop_size[i] - self.input_size[i]) < 3:
                crop_size[i] = self.input_size[i]

        crop_w, crop_h = crop_size

        if self.random_crop:
            x_offset = random.randint(0, img_w - crop_w)
            y_offset = random.randint(0, img_h - crop_h)
        else:
            w_step = (img_w - crop_w) // 4
            h_step = (img_h - crop_h) // 4
            candidate_offsets = [
                (0, 0),  # upper left
                (4 * w_step, 0),  # upper right
                (0, 4 * h_step),  # lower left
                (4 * w_step, 4 * h_step),  # lower right
                (2 * w_step, 2 * h_step),  # center
            ]
            if self.num_fixed_crops == 13:
                extra_candidate_offsets = [
                    (0, 2 * h_step),  # center left
                    (4 * w_step, 2 * h_step),  # center right
                    (2 * w_step, 4 * h_step),  # lower center
                    (2 * w_step, 0 * h_step),  # upper center
                    (1 * w_step, 1 * h_step),  # upper left quarter
                    (3 * w_step, 1 * h_step),  # upper right quarter
                    (1 * w_step, 3 * h_step),  # lower left quarter
                    (3 * w_step, 3 * h_step)  # lower right quarter
                ]
                candidate_offsets.extend(extra_candidate_offsets)
            x_offset, y_offset = random.choice(candidate_offsets)

        new_h, new_w = crop_h, crop_w

        results['crop_bbox'] = np.array(
            [x_offset, y_offset, x_offset + new_w, y_offset + new_h])
        results['img_shape'] = (new_h, new_w)
        results['scales'] = self.scales

        if not self.lazy:
            results['imgs'] = [
                img[y_offset:y_offset + new_h, x_offset:x_offset + new_w]
                for img in results['imgs']
            ]
        else:
            lazyop = results['lazy']
            if lazyop['flip']:
                raise NotImplementedError('Put Flip at last for now')

            # record crop_bbox in lazyop dict to ensure only crop once in Fuse
            lazy_left, lazy_top, lazy_right, lazy_bottom = lazyop['crop_bbox']
            left = x_offset * (lazy_right - lazy_left) / img_w
            right = (x_offset + new_w) * (lazy_right - lazy_left) / img_w
            top = y_offset * (lazy_bottom - lazy_top) / img_h
            bottom = (y_offset + new_h) * (lazy_bottom - lazy_top) / img_h
            lazyop['crop_bbox'] = np.array([(lazy_left + left),
                                            (lazy_top + top),
                                            (lazy_left + right),
                                            (lazy_top + bottom)],
                                           dtype=np.float32)

        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'input_size={self.input_size}, scales={self.scales}, '
                    f'max_wh_scale_gap={self.max_wh_scale_gap}, '
                    f'random_crop={self.random_crop}, '
                    f'num_fixed_crops={self.num_fixed_crops}, '
                    f'lazy={self.lazy})')
        return repr_str


@PIPELINES.register_module()
class Resize(object):
    """Resize images to a specific size.

    Required keys are "imgs", "img_shape", "modality", added or modified
    keys are "imgs", "img_shape", "keep_ratio", "scale_factor", "lazy",
    "resize_size". Required keys in "lazy" is None, added or modified key is
    "interpolation".

    Args:
        scale (float | Tuple[int]): If keep_ratio is True, it serves as scaling
            factor or maximum size:
            If it is a float number, the image will be rescaled by this
            factor, else if it is a tuple of 2 integers, the image will
            be rescaled as large as possible within the scale.
            Otherwise, it serves as (w, h) of output size.
        keep_ratio (bool): If set to True, Images will be resized without
            changing the aspect ratio. Otherwise, it will resize images to a
            given size. Default: True.
        interpolation (str): Algorithm used for interpolation:
            "nearest" | "bilinear". Default: "bilinear".
        lazy (bool): Determine whether to apply lazy operation. Default: False.
    """

    def __init__(self,
                 scale,
                 keep_ratio=True,
                 interpolation='bilinear',
                 lazy=False):
        if isinstance(scale, float):
            if scale <= 0:
                raise ValueError(f'Invalid scale {scale}, must be positive.')
        elif isinstance(scale, tuple):
            max_long_edge = max(scale)
            max_short_edge = min(scale)
            if max_short_edge == -1:
                # assign np.inf to long edge for rescaling short edge later.
                scale = (np.inf, max_long_edge)
        else:
            raise TypeError(
                f'Scale must be float or tuple of int, but got {type(scale)}')
        self.scale = scale
        self.keep_ratio = keep_ratio
        self.interpolation = interpolation
        self.lazy = lazy

    def __call__(self, results):
        """Performs the Resize augmentation.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """

        _init_lazy_if_proper(results, self.lazy)

        if 'scale_factor' not in results:
            results['scale_factor'] = np.array([1, 1], dtype=np.float32)
        img_h, img_w = results['img_shape']

        if self.keep_ratio:
            new_w, new_h = mmcv.rescale_size((img_w, img_h), self.scale)
        else:
            new_w, new_h = self.scale

        self.scale_factor = np.array([new_w / img_w, new_h / img_h],
                                     dtype=np.float32)

        results['img_shape'] = (new_h, new_w)
        results['keep_ratio'] = self.keep_ratio
        results['scale_factor'] = results['scale_factor'] * self.scale_factor

        if not self.lazy:
            results['imgs'] = [
                mmcv.imresize(
                    img, (new_w, new_h), interpolation=self.interpolation)
                for img in results['imgs']
            ]
            if 'grids' in results:
                results['grids'] = [
                    mmcv.imresize(
                        grid, (new_w, new_h), interpolation=self.interpolation)
                    for grid in results['grids']
                ]

        else:
            lazyop = results['lazy']
            if lazyop['flip']:
                raise NotImplementedError('Put Flip at last for now')
            lazyop['interpolation'] = self.interpolation

        if 'ref_seg_map' in results:
            if results['ref_seg_map'].dtype == np.uint8:
                results['ref_seg_map'] = mmcv.imresize(
                    results['ref_seg_map'], (new_w, new_h),
                    interpolation='nearest',
                    backend='pillow')
            else:
                results['ref_seg_map'] = mmcv.imresize(
                    results['ref_seg_map'], (new_w, new_h),
                    interpolation='bilinear',
                    backend='cv2')

        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'scale={self.scale}, keep_ratio={self.keep_ratio}, '
                    f'interpolation={self.interpolation}, '
                    f'lazy={self.lazy})')
        return repr_str


@PIPELINES.register_module()
class Flip(object):
    """Flip the input images with a probability.

    Reverse the order of elements in the given imgs with a specific direction.
    The shape of the imgs is preserved, but the elements are reordered.
    Required keys are "imgs", "img_shape", "modality", added or modified
    keys are "imgs", "lazy" and "flip_direction". Required keys in "lazy" is
    None, added or modified key are "flip" and "flip_direction".

    Args:
        flip_ratio (float): Probability of implementing flip. Default: 0.5.
        direction (str): Flip imgs horizontally or vertically. Options are
            "horizontal" | "vertical". Default: "horizontal".
        lazy (bool): Determine whether to apply lazy operation. Default: False.
    """
    _directions = ['horizontal', 'vertical']

    def __init__(self,
                 flip_ratio=0.5,
                 direction='horizontal',
                 lazy=False,
                 same_on_clip=True,
                 same_across_clip=True,
                 same_clip_indices=None,
                 same_frame_indices=None):
        if direction not in self._directions:
            raise ValueError(f'Direction {direction} is not supported. '
                             f'Currently support ones are {self._directions}')
        self.flip_ratio = flip_ratio
        self.direction = direction
        self.lazy = lazy
        self.same_on_clip = same_on_clip
        self.same_across_clip = same_across_clip
        if same_clip_indices is not None:
            assert isinstance(same_clip_indices, Sequence)
        self.same_clip_indices = same_clip_indices
        if same_frame_indices is not None:
            assert isinstance(same_frame_indices, Sequence)
        self.same_frame_indices = same_frame_indices

    def __call__(self, results):
        """Performs the Flip augmentation.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        _init_lazy_if_proper(results, self.lazy)
        modality = results['modality']
        if modality == 'Flow':
            assert self.direction == 'horizontal'

        if np.random.rand() < self.flip_ratio:
            flip = True
        else:
            flip = False

        results['flip'] = flip
        results['flip_direction'] = self.direction

        if not self.lazy:
            for i, img in enumerate(results['imgs']):
                is_new_clip = not self.same_across_clip and i % results[
                    'clip_len'] == 0 and i > 0
                generate_new = not self.same_on_clip or is_new_clip
                if self.same_clip_indices is not None:
                    assert min(self.same_clip_indices) >= 0
                    assert max(self.same_clip_indices) < results['num_clips']
                    keep_same = i % results[
                        'num_clips'] in self.same_clip_indices
                    generate_new = generate_new and not keep_same
                if self.same_frame_indices is not None:
                    assert min(self.same_frame_indices) >= 0
                    assert max(self.same_frame_indices) < results['clip_len']
                    keep_same = i % results[
                        'clip_len'] in self.same_frame_indices
                    generate_new = generate_new and not keep_same
                if generate_new:
                    flip = npr.rand() < self.flip_ratio
                if flip:
                    mmcv.imflip_(img, self.direction)
                    if 'grids' in results:
                        mmcv.imflip_(results['grids'][i], self.direction)
            if flip:
                lt = len(results['imgs'])
                for i in range(0, lt, 2):
                    # flow with even indexes are x_flow, which need to be
                    # inverted when doing horizontal flip
                    if modality == 'Flow':
                        results['imgs'][i] = mmcv.iminvert(results['imgs'][i])

            else:
                results['imgs'] = list(results['imgs'])
        else:
            lazyop = results['lazy']
            if lazyop['flip']:
                raise NotImplementedError('Use one Flip please')
            lazyop['flip'] = flip
            lazyop['flip_direction'] = self.direction

        return results

    def __repr__(self):
        repr_str = (
            f'{self.__class__.__name__}('
            f'flip_ratio={self.flip_ratio}, direction={self.direction}, '
            f'lazy={self.lazy})')
        return repr_str


@PIPELINES.register_module()
class Normalize(object):
    """Normalize images with the given mean and std value.

    Required keys are "imgs", "img_shape", "modality", added or modified
    keys are "imgs" and "img_norm_cfg". If modality is 'Flow', additional
    keys "scale_factor" is required

    Args:
        mean (Sequence[float]): Mean values of different channels.
        std (Sequence[float]): Std values of different channels.
        to_bgr (bool): Whether to convert channels from RGB to BGR.
            Default: False.
        adjust_magnitude (bool): Indicate whether to adjust the flow magnitude
            on 'scale_factor' when modality is 'Flow'. Default: False.
    """

    def __init__(self, mean, std, to_bgr=False, adjust_magnitude=False):
        if not isinstance(mean, Sequence):
            raise TypeError(
                f'Mean must be list, tuple or np.ndarray, but got {type(mean)}'
            )

        if not isinstance(std, Sequence):
            raise TypeError(
                f'Std must be list, tuple or np.ndarray, but got {type(std)}')

        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.to_bgr = to_bgr
        self.adjust_magnitude = adjust_magnitude

    def __call__(self, results):
        modality = results['modality']

        if modality == 'RGB':
            n = len(results['imgs'])
            h, w, c = results['imgs'][0].shape
            imgs = np.empty((n, h, w, c), dtype=np.float32)
            for i, img in enumerate(results['imgs']):
                imgs[i] = img

            for img in imgs:
                mmcv.imnormalize_(img, self.mean, self.std, self.to_bgr)

            results['imgs'] = imgs
            results['img_norm_cfg'] = dict(
                mean=self.mean, std=self.std, to_bgr=self.to_bgr)
            return results
        elif modality == 'Flow':
            num_imgs = len(results['imgs'])
            assert num_imgs % 2 == 0
            assert self.mean.shape[0] == 2
            assert self.std.shape[0] == 2
            n = num_imgs // 2
            h, w = results['imgs'][0].shape
            x_flow = np.empty((n, h, w), dtype=np.float32)
            y_flow = np.empty((n, h, w), dtype=np.float32)
            for i in range(n):
                x_flow[i] = results['imgs'][2 * i]
                y_flow[i] = results['imgs'][2 * i + 1]
            x_flow = (x_flow - self.mean[0]) / self.std[0]
            y_flow = (y_flow - self.mean[1]) / self.std[1]
            if self.adjust_magnitude:
                x_flow = x_flow * results['scale_factor'][0]
                y_flow = y_flow * results['scale_factor'][1]
            imgs = np.stack([x_flow, y_flow], axis=-1)
            results['imgs'] = imgs
            args = dict(
                mean=self.mean,
                std=self.std,
                to_bgr=self.to_bgr,
                adjust_magnitude=self.adjust_magnitude)
            results['img_norm_cfg'] = args
            return results
        else:
            raise NotImplementedError

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'mean={self.mean}, '
                    f'std={self.std}, '
                    f'to_bgr={self.to_bgr}, '
                    f'adjust_magnitude={self.adjust_magnitude})')
        return repr_str


@PIPELINES.register_module()
class CenterCrop(object):
    """Crop the center area from images.

    Required keys are "imgs", "img_shape", added or modified keys are "imgs",
    "crop_bbox", "lazy" and "img_shape". Required keys in "lazy" is
    "crop_bbox", added or modified key is "crop_bbox".

    Args:
        crop_size (int | tuple[int]): (w, h) of crop size.
        lazy (bool): Determine whether to apply lazy operation. Default: False.
    """

    def __init__(self, crop_size, lazy=False):
        self.crop_size = _pair(crop_size)
        self.lazy = lazy
        if not mmcv.is_tuple_of(self.crop_size, int):
            raise TypeError(f'Crop_size must be int or tuple of int, '
                            f'but got {type(crop_size)}')

    def __call__(self, results):
        """Performs the CenterCrop augmentation.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        _init_lazy_if_proper(results, self.lazy)

        img_h, img_w = results['img_shape']
        crop_w, crop_h = self.crop_size

        left = (img_w - crop_w) // 2
        top = (img_h - crop_h) // 2
        right = left + crop_w
        bottom = top + crop_h
        new_h, new_w = bottom - top, right - left

        results['crop_bbox'] = np.array([left, top, right, bottom])
        results['img_shape'] = (new_h, new_w)

        if not self.lazy:
            results['imgs'] = [
                img[top:bottom, left:right] for img in results['imgs']
            ]
        else:
            lazyop = results['lazy']
            if lazyop['flip']:
                raise NotImplementedError('Put Flip at last for now')

            # record crop_bbox in lazyop dict to ensure only crop once in Fuse
            lazy_left, lazy_top, lazy_right, lazy_bottom = lazyop['crop_bbox']
            left = left * (lazy_right - lazy_left) / img_w
            right = right * (lazy_right - lazy_left) / img_w
            top = top * (lazy_bottom - lazy_top) / img_h
            bottom = bottom * (lazy_bottom - lazy_top) / img_h
            lazyop['crop_bbox'] = np.array([(lazy_left + left),
                                            (lazy_top + top),
                                            (lazy_left + right),
                                            (lazy_top + bottom)],
                                           dtype=np.float32)

        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}(crop_size={self.crop_size}, '
                    f'lazy={self.lazy})')
        return repr_str


@PIPELINES.register_module()
class ThreeCrop(object):
    """Crop images into three crops.

    Crop the images equally into three crops with equal intervals along the
    shorter side.
    Required keys are "imgs", "img_shape", added or modified keys are "imgs",
    "crop_bbox" and "img_shape".

    Args:
        crop_size(int | tuple[int]): (w, h) of crop size.
    """

    def __init__(self, crop_size):
        self.crop_size = _pair(crop_size)
        if not mmcv.is_tuple_of(self.crop_size, int):
            raise TypeError(f'Crop_size must be int or tuple of int, '
                            f'but got {type(crop_size)}')

    def __call__(self, results):
        """Performs the ThreeCrop augmentation.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        _init_lazy_if_proper(results, False)

        imgs = results['imgs']
        img_h, img_w = results['imgs'][0].shape[:2]
        crop_w, crop_h = self.crop_size
        assert crop_h == img_h or crop_w == img_w

        if crop_h == img_h:
            w_step = (img_w - crop_w) // 2
            offsets = [
                (0, 0),  # left
                (2 * w_step, 0),  # right
                (w_step, 0),  # middle
            ]
        elif crop_w == img_w:
            h_step = (img_h - crop_h) // 2
            offsets = [
                (0, 0),  # top
                (0, 2 * h_step),  # down
                (0, h_step),  # middle
            ]

        cropped = []
        crop_bboxes = []
        for x_offset, y_offset in offsets:
            bbox = [x_offset, y_offset, x_offset + crop_w, y_offset + crop_h]
            crop = [
                img[y_offset:y_offset + crop_h, x_offset:x_offset + crop_w]
                for img in imgs
            ]
            cropped.extend(crop)
            crop_bboxes.extend([bbox for _ in range(len(imgs))])

        crop_bboxes = np.array(crop_bboxes)
        results['imgs'] = cropped
        results['crop_bbox'] = crop_bboxes
        results['img_shape'] = results['imgs'][0].shape[:2]

        return results

    def __repr__(self):
        repr_str = f'{self.__class__.__name__}(crop_size={self.crop_size})'
        return repr_str


@PIPELINES.register_module()
class TenCrop(object):
    """Crop the images into 10 crops (corner + center + flip).

    Crop the four corners and the center part of the image with the same
    given crop_size, and flip it horizontally.
    Required keys are "imgs", "img_shape", added or modified keys are "imgs",
    "crop_bbox" and "img_shape".

    Args:
        crop_size(int | tuple[int]): (w, h) of crop size.
    """

    def __init__(self, crop_size):
        self.crop_size = _pair(crop_size)
        if not mmcv.is_tuple_of(self.crop_size, int):
            raise TypeError(f'Crop_size must be int or tuple of int, '
                            f'but got {type(crop_size)}')

    def __call__(self, results):
        """Performs the TenCrop augmentation.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        _init_lazy_if_proper(results, False)

        imgs = results['imgs']

        img_h, img_w = results['imgs'][0].shape[:2]
        crop_w, crop_h = self.crop_size

        w_step = (img_w - crop_w) // 4
        h_step = (img_h - crop_h) // 4

        offsets = [
            (0, 0),  # upper left
            (4 * w_step, 0),  # upper right
            (0, 4 * h_step),  # lower left
            (4 * w_step, 4 * h_step),  # lower right
            (2 * w_step, 2 * h_step),  # center
        ]

        img_crops = list()
        crop_bboxes = list()
        for x_offset, y_offsets in offsets:
            crop = [
                img[y_offsets:y_offsets + crop_h, x_offset:x_offset + crop_w]
                for img in imgs
            ]
            flip_crop = [np.flip(c, axis=1).copy() for c in crop]
            bbox = [x_offset, y_offsets, x_offset + crop_w, y_offsets + crop_h]
            img_crops.extend(crop)
            img_crops.extend(flip_crop)
            crop_bboxes.extend([bbox for _ in range(len(imgs) * 2)])

        crop_bboxes = np.array(crop_bboxes)
        results['imgs'] = img_crops
        results['crop_bbox'] = crop_bboxes
        results['img_shape'] = results['imgs'][0].shape[:2]

        return results

    def __repr__(self):
        repr_str = f'{self.__class__.__name__}(crop_size={self.crop_size})'
        return repr_str


@PIPELINES.register_module()
class MultiGroupCrop(object):
    """Randomly crop the images into several groups.

    Crop the random region with the same given crop_size and bounding box
    into several groups.
    Required keys are "imgs", added or modified keys are "imgs", "crop_bbox"
    and "img_shape".

    Args:
        crop_size(int | tuple[int]): (w, h) of crop size.
        groups(int): Number of groups.
    """

    def __init__(self, crop_size, groups):
        self.crop_size = _pair(crop_size)
        self.groups = groups
        if not mmcv.is_tuple_of(self.crop_size, int):
            raise TypeError(
                'Crop size must be int or tuple of int, but got {}'.format(
                    type(crop_size)))

        if not isinstance(groups, int):
            raise TypeError(f'Groups must be int, but got {type(groups)}.')

        if groups <= 0:
            raise ValueError('Groups must be positive.')

    def __call__(self, results):
        """Performs the MultiGroupCrop augmentation.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        imgs = results['imgs']
        img_h, img_w = imgs[0].shape[:2]
        crop_w, crop_h = self.crop_size

        img_crops = []
        crop_bboxes = []
        for _ in range(self.groups):
            x_offset = random.randint(0, img_w - crop_w)
            y_offset = random.randint(0, img_h - crop_h)

            bbox = [x_offset, y_offset, x_offset + crop_w, y_offset + crop_h]
            crop = [
                img[y_offset:y_offset + crop_h, x_offset:x_offset + crop_w]
                for img in imgs
            ]
            img_crops.extend(crop)
            crop_bboxes.extend([bbox for _ in range(len(imgs))])

        crop_bboxes = np.array(crop_bboxes)
        results['imgs'] = img_crops
        results['crop_bbox'] = crop_bboxes
        results['img_shape'] = results['imgs'][0].shape[:2]

        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}'
                    f'(crop_size={self.crop_size}, '
                    f'groups={self.groups})')
        return repr_str


@PIPELINES.register_module()
class RGB2LAB(object):

    def __init__(self):
        pass

    def __call__(self, results):
        for i, img in enumerate(results['imgs']):
            results['imgs'][i] = mmcv.imconvert(img, 'rgb', 'lab')
        return results


@PIPELINES.register_module()
class PhotoMetricDistortion(object):
    """Apply photometric distortion to image sequentially, every transformation
    is applied with a probability of 0.5. The position of random contrast is in
    second or second to last.

    1. random brightness
    2. random contrast (mode 0)
    3. convert color from BGR to HSV
    4. random saturation
    5. random hue
    6. convert color from HSV to BGR
    7. random contrast (mode 1)
    8. randomly swap channels

    Args:
        brightness_delta (int): delta of brightness.
        contrast_range (tuple): range of contrast.
        saturation_range (tuple): range of saturation.
        hue_delta (int): delta of hue.
    """

    def __init__(self,
                 brightness_delta=32,
                 contrast_range=(0.5, 1.5),
                 saturation_range=(0.5, 1.5),
                 hue_delta=18,
                 p=0.5,
                 same_on_clip=True,
                 same_across_clip=True):
        self.brightness_delta = brightness_delta
        self.contrast_lower, self.contrast_upper = contrast_range
        self.saturation_lower, self.saturation_upper = saturation_range
        self.hue_delta = hue_delta
        self.p = p
        self.same_on_clip = same_on_clip
        self.same_across_clip = same_across_clip

    def convert(self, img, alpha=1, beta=0):
        """Multiple with alpha and add beat with clip."""
        img = img.astype(np.float32) * alpha + beta
        img = np.clip(img, 0, 255)
        return img.astype(np.uint8)

    def brightness(self, img, beta):
        """Brightness distortion."""
        return self.convert(img, beta=beta)

    def contrast(self, img, alpha):
        """Contrast distortion."""
        return self.convert(img, alpha=alpha)

    def saturation(self, img, alpha):
        """Saturation distortion."""
        img = mmcv.bgr2hsv(img)
        img[:, :, 1] = self.convert(img[:, :, 1], alpha=alpha)
        img = mmcv.hsv2bgr(img)
        return img

    def hue(self, img, delta):
        """Hue distortion."""
        img = mmcv.bgr2hsv(img)
        img[:, :, 0] = (img[:, :, 0].astype(int) + delta) % 180
        img = mmcv.hsv2bgr(img)
        return img

    def __call__(self, results):
        """Call function to perform photometric distortion on images.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Result dict with images distorted.
        """

        apply_bright = npr.rand() < self.p
        bright_beta = npr.uniform(-self.brightness_delta,
                                  self.brightness_delta)
        apply_contrast = npr.rand() < self.p
        contrast_alpha = npr.uniform(self.contrast_lower, self.contrast_upper)
        apply_saturation = npr.rand() < self.p
        saturation_alpha = npr.uniform(self.saturation_lower,
                                       self.saturation_upper)
        apply_hue = npr.rand() < self.p
        hue_delta = npr.randint(-self.hue_delta, self.hue_delta)
        apply_mode = npr.rand() < self.p

        for i, img in enumerate(results['imgs']):
            is_new_clip = not self.same_across_clip and i % results[
                'clip_len'] == 0 and i > 0
            if not self.same_on_clip or is_new_clip:
                apply_bright = npr.rand() < self.p
                bright_beta = npr.uniform(-self.brightness_delta,
                                          self.brightness_delta)
                apply_contrast = npr.rand() < self.p
                contrast_alpha = npr.uniform(self.contrast_lower,
                                             self.contrast_upper)
                apply_saturation = npr.rand() < self.p
                saturation_alpha = npr.uniform(self.saturation_lower,
                                               self.saturation_upper)
                apply_hue = npr.rand() < self.p
                hue_delta = npr.randint(-self.hue_delta, self.hue_delta)
                apply_mode = npr.rand() < self.p
            # random brightness
            if apply_bright:
                img = self.brightness(img, beta=bright_beta)

            if apply_mode and apply_contrast:
                img = self.contrast(img, alpha=contrast_alpha)

            # random saturation
            if apply_saturation:
                img = self.saturation(img, alpha=saturation_alpha)

            # random hue
            if apply_hue:
                img = self.hue(img, delta=hue_delta)

            # random contrast
            if not apply_mode and apply_contrast:
                img = self.contrast(img, alpha=saturation_alpha)

            results['imgs'][i] = img
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += (f'(brightness_delta={self.brightness_delta}, '
                     f'contrast_range=({self.contrast_lower}, '
                     f'{self.contrast_upper}), '
                     f'saturation_range=({self.saturation_lower}, '
                     f'{self.saturation_upper}), '
                     f'hue_delta={self.hue_delta})')
        return repr_str


@PIPELINES.register_module()
class RandomGaussianBlur(object):

    def __init__(self,
                 sigma_range=(0.1, 0.2),
                 p=0.5,
                 same_on_clip=True,
                 same_across_clip=True):
        self.sigma_range = sigma_range
        self.p = p
        self.same_on_clip = same_on_clip
        self.same_across_clip = same_across_clip

    def __call__(self, results):
        apply = npr.rand() < self.p
        sigma = random.uniform(self.sigma_range[0], self.sigma_range[1])
        for i, img in enumerate(results['imgs']):
            is_new_clip = not self.same_across_clip and i % results[
                'clip_len'] == 0 and i > 0
            if not self.same_on_clip or is_new_clip:
                apply = npr.rand() < self.p
                sigma = random.uniform(self.sigma_range[0],
                                       self.sigma_range[1])
            if apply:
                pil_image = Image.fromarray(img)
                pil_image = pil_image.filter(
                    ImageFilter.GaussianBlur(radius=sigma))
                img = np.array(pil_image)
                results['imgs'][i] = img

        return results


@PIPELINES.register_module()
class RandomGrayScale(object):

    def __init__(self, p=0.5, same_on_clip=True, same_across_clip=True):
        self.p = p
        self.same_on_clip = same_on_clip
        self.same_across_clip = same_across_clip

    def __call__(self, results):
        apply = npr.rand() < self.p
        for i, img in enumerate(results['imgs']):
            is_new_clip = not self.same_across_clip and i % results[
                'clip_len'] == 0 and i > 0
            if not self.same_on_clip or is_new_clip:
                apply = npr.rand() < self.p
            if apply:
                img = mmcv.rgb2gray(img, keepdim=True)
                img = np.repeat(img, 3, axis=-1)
                results['imgs'][i] = img

        return results


@PIPELINES.register_module()
class ColorJitter(object):

    def __init__(self,
                 p=0.5,
                 same_on_clip=True,
                 same_across_clip=True,
                 brightness=0,
                 contrast=0,
                 saturation=0,
                 hue=0):
        trans = _ColorJitter(
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            hue=hue)
        self.brightness = trans.brightness
        self.contrast = trans.contrast
        self.saturation = trans.saturation
        self.hue = trans.hue
        self.p = p
        self.same_on_clip = same_on_clip
        self.same_across_clip = same_across_clip

    def __call__(self, results):
        apply = npr.rand() < self.p
        trans = _ColorJitter.get_params(self.brightness, self.contrast,
                                        self.saturation, self.hue)
        for i, img in enumerate(results['imgs']):
            is_new_clip = not self.same_across_clip and i % results[
                'clip_len'] == 0 and i > 0
            if not self.same_on_clip or is_new_clip:
                apply = npr.rand() < self.p
                trans = _ColorJitter.get_params(self.brightness, self.contrast,
                                                self.saturation, self.hue)
            if apply:
                img = np.array(trans(Image.fromarray(img)))
                results['imgs'][i] = img

        return results


@PIPELINES.register_module()
class Grid(object):

    def __init__(self, normalize=False):
        self.normalize = normalize

    def __call__(self, results):
        h, w = results['original_shape']
        y_grid, x_grid = np.meshgrid(
            range(h), range(w), indexing='ij', sparse=False)
        if self.normalize:
            y_grid = 2 * y_grid / h - 1
            x_grid = 2 * x_grid / w - 1
        grids = [
            np.stack((y_grid, x_grid), axis=-1).astype(np.float)
            for _ in range(len(results['imgs']))
        ]

        results['grids'] = grids

        return results


# TODO not tested
@PIPELINES.register_module()
class Image2Patch(object):

    def __init__(self, patch_size, stride, scale_jitter=(0.7, 0.9)):
        self.patch_size = patch_size
        self.stride = stride
        self.crop_trans = _RandomResizedCrop(patch_size, scale=scale_jitter)

    def __call__(self, results):

        patches = []
        for img in results['imgs']:
            patch = view_as_windows(img, self.patch_size, self.stride)
            patches.extend(list(patch.view(-1, *patch.shape[2:])))
        for i in range(len(patches)):
            patches[i] = self.crop_trans(patches[i])
        results['imgs'] = patches

        return results


@PIPELINES.register_module()
class HidePatch(object):
    """after normalization."""

    def __init__(self, patch_size, hide_prob):
        if not isinstance(patch_size, (list, tuple)):
            patch_size = [patch_size]
        self.patch_size = patch_size
        self.hide_prob = hide_prob

    def __call__(self, results):
        patch_size = np.random.choice(self.patch_size)
        h, w = results['imgs'][0].shape[:2]
        for i, img in enumerate(results['imgs']):
            for y in range(0, h, patch_size):
                for x in range(0, w, patch_size):
                    apply = npr.rand() < self.hide_prob
                    if apply:
                        results['imgs'][i][y:y + patch_size,
                                           x:x + patch_size] = 0

        return results


@PIPELINES.register_module()
class RandomAffine(object):

    def __init__(self,
                 degrees,
                 p=0.5,
                 same_on_clip=True,
                 same_across_clip=True,
                 translate=None,
                 scale=None,
                 shear=None,
                 resample=2,
                 fillcolor=0):
        trans = _RandomAffine(
            degrees=degrees,
            translate=translate,
            scale=scale,
            shear=shear,
            resample=resample,
            fillcolor=fillcolor)
        self.degrees = trans.degrees
        self.translate = trans.translate
        self.scale = trans.scale
        self.shear = trans.shear
        self.resample = trans.resample
        self.fillcolor = trans.fillcolor
        self.p = p
        self.same_on_clip = same_on_clip
        self.same_across_clip = same_across_clip

    def __call__(self, results):
        apply = npr.rand() < self.p
        h, w = results['imgs'][0].shape[:2]
        ret = _RandomAffine.get_params(self.degrees, self.translate,
                                       self.scale, self.shear, (w, h))
        for i, img in enumerate(results['imgs']):
            is_new_clip = not self.same_across_clip and i % results[
                'clip_len'] == 0 and i > 0
            if not self.same_on_clip or is_new_clip:
                apply = npr.rand() < self.p
                ret = _RandomAffine.get_params(self.degrees, self.translate,
                                               self.scale, self.shear, (w, h))
            if apply:
                img = np.array(
                    F.affine(
                        Image.fromarray(img),
                        *ret,
                        resample=self.resample,
                        fillcolor=self.fillcolor))
                results['imgs'][i] = img

        return results


@PIPELINES.register_module()
class RandomChoiceRotate(object):

    def __init__(self, p, degrees, same_on_clip=True, same_across_clip=True):
        self.p = p
        if not isinstance(degrees, (list, tuple)):
            degrees = [degrees]
        self.degrees = degrees
        self.label_map = {d: i for i, d in enumerate(degrees)}
        self.same_on_clip = same_on_clip
        self.same_across_clip = same_across_clip

    def __call__(self, results):
        apply = npr.rand() < self.p
        degree = np.random.choice(self.degrees)
        labels = []
        for i, img in enumerate(results['imgs']):
            is_new_clip = not self.same_across_clip and i % results[
                'clip_len'] == 0 and i > 0
            if not self.same_on_clip or is_new_clip:
                apply = npr.rand() < self.p
                degree = np.random.choice(self.degrees)
            if apply:
                img = np.array(mmcv.imrotate(img, angle=degree))
                results['imgs'][i] = img
                labels.append(self.label_map[degree])
            else:
                labels.append(0)
        results['rotation_labels'] = np.array(labels)

        return results


@PIPELINES.register_module()
class RandomErasing(object):
    """ Randomly selects a rectangle region in an image and erases its pixels.
        'Random Erasing Data Augmentation' by Zhong et al.
        See https://arxiv.org/pdf/1708.04896.pdf
        This variant of RandomErasing is intended to be applied to either a
        batch or single image tensor after it has been normalized by dataset
        mean and std.
    Args:
         p: Probability that the Random Erasing operation will be performed.
         min_area: Minimum percentage of erased area wrt input image area.
         max_area: Maximum percentage of erased area wrt input image area.
         min_aspect: Minimum aspect ratio of erased area.
         mode: pixel color mode, one of 'const', 'rand', or 'pixel'
            'const' - erase block is constant color of 0 for all channels
            'rand'  - erase block is same per-channel random (normal) color
            'pixel' - erase block is per-pixel random (normal) color
        max_count: maximum number of erasing blocks per image, area per box is
            scaled by count. per-image count is randomly chosen between 1 and
            this value.
    """

    def __init__(self,
                 p=0.5,
                 area_range=(0.02, 1 / 3),
                 aspect_ratio_range=(1 / 3, 3),
                 count_range=(1, 1),
                 mode='const'):
        self.p = p
        self.area_range = area_range
        self.aspect_ratio_range = aspect_ratio_range
        self.count_range = count_range
        assert mode in ['rand', 'pixel', 'const']
        self.mode = mode

    @staticmethod
    def get_crop_bbox(img_shape,
                      area_range,
                      aspect_ratio_range,
                      max_attempts=10):
        """Get a crop bbox given the area range and aspect ratio range.

        Args:
            img_shape (Tuple[int]): Image shape
            area_range (Tuple[float]): The candidate area scales range of
                output cropped images. Default: (0.08, 1.0).
            aspect_ratio_range (Tuple[float]): The candidate aspect
                ratio range of output cropped images. Default: (3 / 4, 4 / 3).
                max_attempts (int): The maximum of attempts. Default: 10.
            max_attempts (int): Max attempts times to generate random candidate
                bounding box. If it doesn't qualified one, the center bounding
                box will be used.
        Returns:
            (list[int]) A random crop bbox within the area range and aspect
            ratio range.
        """
        assert 0 < area_range[0] <= area_range[1] <= 1
        assert 0 < aspect_ratio_range[0] <= aspect_ratio_range[1]

        img_h, img_w = img_shape
        area = img_h * img_w

        min_ar, max_ar = aspect_ratio_range
        aspect_ratios = np.exp(
            np.random.uniform(
                np.log(min_ar), np.log(max_ar), size=max_attempts))
        target_areas = np.random.uniform(*area_range, size=max_attempts) * area
        candidate_crop_w = np.round(np.sqrt(target_areas *
                                            aspect_ratios)).astype(np.int32)
        candidate_crop_h = np.round(np.sqrt(target_areas /
                                            aspect_ratios)).astype(np.int32)

        for i in range(max_attempts):
            crop_w = candidate_crop_w[i]
            crop_h = candidate_crop_h[i]
            if crop_h <= img_h and crop_w <= img_w:
                x_offset = random.randint(0, img_w - crop_w)
                y_offset = random.randint(0, img_h - crop_h)
                return x_offset, y_offset, x_offset + crop_w, y_offset + crop_h

        # Fallback
        crop_size = min(img_h, img_w)
        x_offset = (img_w - crop_size) // 2
        y_offset = (img_h - crop_size) // 2
        return x_offset, y_offset, x_offset + crop_size, y_offset + crop_size

    def get_pixels(self, patch_shape):
        if self.mode == 'pixel':
            return np.random.randn(*patch_shape)
        elif self.mode == 'rand':
            return np.random.randn(1, 1, patch_shape[-1])
        else:
            return np.zeros(patch_shape, dtype=np.float)

    def erase(self, img):
        count = random.randint(*self.count_range)
        img_h, img_w = img.shape[:2]
        for _ in range(count):
            left, top, right, bottom = self.get_crop_bbox(
                (img_h, img_w),
                (self.area_range[0] / count, self.area_range[1] / count),
                self.aspect_ratio_range)
            new_h, new_w = bottom - top, right - left
            img[top:bottom, left:right] = self.get_pixels(
                (new_h, new_w, img.shape[2]))

        return img

    def __call__(self, results):
        for i, img in enumerate(results['imgs']):
            apply = npr.rand() < self.p
            if apply:
                results['imgs'][i] = self.erase(img)

        return results
