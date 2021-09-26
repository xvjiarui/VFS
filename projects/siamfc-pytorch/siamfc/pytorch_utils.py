import numpy as np
import torch
from PIL import Image


def _is_pil_image(img):
    return isinstance(img, Image.Image)


def _is_tensor_image(img):
    return torch.is_tensor(img) and img.ndimension() == 3


def _is_numpy_image(img):
    return isinstance(img, np.ndarray) and (img.ndim in {2, 3})


def from_numpy(np_array):
    global from_numpy_warn
    if isinstance(np_array, list) or isinstance(np_array, tuple):
        try:
            np_array = np.stack(np_array, 0)
        except ValueError:
            np_array = np.stack([from_numpy(val) for val in np_array], 0)
    elif isinstance(np_array, dict):
        return {key: from_numpy(val) for key, val in np_array.items()}
    np_array = np.asarray(np_array)
    if np_array.dtype == np.uint32:
        if not from_numpy_warn[np.uint32]:
            print('numpy -> torch dtype uint32 not supported, using int32')
            from_numpy_warn[np.uint32] = True
        np_array = np_array.astype(np.int32)
    elif np_array.dtype == np.dtype('O'):
        if not from_numpy_warn[np.dtype('O')]:
            print('numpy -> torch dtype Object not supported, '
                  'returning numpy array')
            from_numpy_warn[np.dtype('O')] = True
        return np_array
    elif np_array.dtype.type == np.str_:
        if not from_numpy_warn[np.str_]:
            print('numpy -> torch dtype numpy.str_ not supported, '
                  'returning numpy array')
            from_numpy_warn[np.str_] = True
        return np_array
    return torch.from_numpy(np_array)


class ToTensor(object):

    def __init__(self, transpose=True, scale=255):
        self.transpose = transpose
        self.scale = scale

    def __call__(self, pic):
        """Convert a ``PIL Image`` or ``numpy.ndarray`` to tensor.

        See ``ToTensor`` for more details.
        Args:
            pic (PIL Image or numpy.ndarray): Image to be converted to tensor.
        Returns:
            Tensor: Converted image.
        """
        if not (_is_pil_image(pic) or _is_numpy_image(pic)):
            raise TypeError(
                'pic should be PIL Image or ndarray. Got {}'.format(type(pic)))

        if isinstance(pic, np.ndarray):
            # handle numpy array
            if pic.ndim == 2:
                pic = pic[:, :, None]

            if self.transpose:
                pic = pic.transpose((2, 0, 1))
            img = torch.from_numpy(pic)
            if self.scale is not None:
                return img.float().div(self.scale)
            return img

        # handle PIL Image
        if pic.mode == 'I':
            img = torch.from_numpy(np.array(pic, np.int32, copy=False))
        elif pic.mode == 'I;16':
            img = torch.from_numpy(np.array(pic, np.int16, copy=False))
        elif pic.mode == 'F':
            img = torch.from_numpy(np.array(pic, np.float32, copy=False))
        elif pic.mode == '1':
            img = 255 * torch.from_numpy(np.array(pic, np.uint8, copy=False))
        else:
            img = torch.ByteTensor(
                torch.ByteStorage.from_buffer(pic.tobytes()))
        # PIL image mode: L, LA, P, I, F, RGB, YCbCr, RGBA, CMYK
        if pic.mode == 'YCbCr':
            nchannel = 3
        elif pic.mode == 'I;16':
            nchannel = 1
        else:
            nchannel = len(pic.mode)
        img = img.view(pic.size[1], pic.size[0], nchannel)
        # put it from HWC to CHW format
        # yikes, this transpose takes 80% of the loading time/CPU
        if self.transpose:
            img = img.permute(2, 0, 1).contiguous()
        if self.scale is not None:
            return img.float().div(self.scale)
        else:
            return img
