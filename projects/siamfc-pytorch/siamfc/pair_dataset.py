import random

import cv2
import numpy as np
from torch.utils.data import Dataset

from .pytorch_utils import from_numpy


def fliplr(x):
    # Copy because needs to be contiguous with positive stride
    return np.fliplr(x).copy()


class PairDataset(Dataset):

    def __init__(self,
                 seqs,
                 data_subset='train',
                 pair_transform=None,
                 transforms=None,
                 pairs_per_seq=25):
        super(PairDataset, self).__init__()
        self.seqs = seqs
        self.data_subset = data_subset
        self.pair_transform = pair_transform
        self.transforms = transforms
        self.pairs_per_seq = pairs_per_seq
        self.indices = np.random.permutation(len(seqs))
        self.length = np.sum(self.indices)
        self.return_meta = getattr(seqs, 'return_meta', False)
        self.seq_sizes = {}
        self.invalid_seqs = {}

    def __getitem__(self, index):
        index = self.indices[index % len(self.indices)]

        # get filename lists and annotations
        if self.return_meta:
            img_files, anno, meta = self.seqs[index]
            vis_ratios = meta.get('cover', None)
        else:
            img_files, anno = self.seqs[index][:2]
            vis_ratios = None
        # filter out noisy frames

        val_indices = self._filter(img_files[0], index, anno, vis_ratios)
        if len(val_indices) < 2:
            index = np.random.choice(len(self))
            return self.__getitem__(index)

        # sample a frame pair
        rand_z, rand_x = self._sample_pair(val_indices)

        box_z = anno[rand_z]
        box_x = anno[rand_x]

        z = cv2.imread(img_files[rand_z])[:, :, ::-1]
        x = cv2.imread(img_files[rand_x])[:, :, ::-1]
        # z = cv2.cvtColor(z, cv2.COLOR_BGR2RGB)
        # x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
        item = (z, x, box_z, box_x)
        if self.pair_transform is not None:
            exemplar_img, track_img = self.pair_transform(item)
            label = None
            if isinstance(track_img, tuple):
                track_img, label = track_img
            if self.transforms is not None:
                if self.data_subset == 'train' and random.random() > 0.5:
                    exemplar_img = np.fliplr(
                        exemplar_img).copy()  # Need to copy to make contiguous
                if self.data_subset == 'train' and random.random() > 0.5:
                    track_img = np.fliplr(track_img).copy()
                    if label is not None:
                        label = np.fliplr(label).copy()
                exemplar_img = self.transforms(exemplar_img)
                track_img = self.transforms(track_img)
            if label is not None:
                item = (exemplar_img, track_img,
                        from_numpy(label[np.newaxis, ...]))
            else:
                item = (exemplar_img, track_img)
        return item

    def __len__(self):
        return len(self.indices) * self.pairs_per_seq

    def _sample_pair(self, indices):
        n = len(indices)
        assert n > 0

        if n == 1:
            return indices[0], indices[0]
        elif n == 2:
            return indices[0], indices[1]
        else:
            for i in range(100):
                rand_z, rand_x = np.sort(
                    np.random.choice(indices, 2, replace=False))
                if rand_x - rand_z < 100:
                    # if 30 < abs(rand_x - rand_z) < 500:
                    break
            else:
                rand_z = np.random.choice(indices)
                rand_x = rand_z

            return rand_z, rand_x

    def _filter(self, img0, key, anno, vis_ratios=None):
        if key in self.invalid_seqs:
            return self.invalid_seqs[key]
        if key not in self.seq_sizes:
            self.seq_sizes[key] = cv2.imread(img0).shape[:2]
        size = self.seq_sizes[key]
        areas = anno[:, 2] * anno[:, 3]

        # acceptance conditions
        c1 = areas >= 20
        c2 = np.all(anno[:, 2:] >= 20, axis=1)
        c3 = np.all(anno[:, 2:] <= 500, axis=1)
        c4 = np.all((anno[:, 2:] / size) >= 0.01, axis=1)
        c5 = np.all((anno[:, 2:] / size) <= 0.5, axis=1)
        c6 = (anno[:, 2] / np.maximum(1, anno[:, 3])) >= 0.25
        c7 = (anno[:, 2] / np.maximum(1, anno[:, 3])) <= 4
        if vis_ratios is not None:
            c8 = vis_ratios > max(1, vis_ratios.max() * 0.3)
        else:
            c8 = np.ones_like(c1)

        mask = np.logical_and.reduce((c1, c2, c3, c4, c5, c6, c7, c8))
        val_indices = np.where(mask)[0]
        if len(val_indices) < 2:
            self.invalid_seqs[key] = val_indices
        return val_indices
