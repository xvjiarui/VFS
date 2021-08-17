import os
import os.path as osp

from torchvision.datasets.folder import IMG_EXTENSIONS, make_dataset

from .builder import DATASETS
from .video_dataset import VideoDataset


@DATASETS.register_module()
class ImageDataset(VideoDataset):

    def __init__(self, ann_file, pipeline, start_index=0, **kwargs):
        super().__init__(ann_file, pipeline, start_index=start_index, **kwargs)

    def _find_classes(self, dir):
        """Finds the class folders in a dataset.

        Args:
            dir (string): Root directory path.

        Returns:
            tuple: (classes, class_to_idx) where classes are relative to
            (dir), and class_to_idx is a dictionary.

        Ensures:
            No class is a subdirectory of another.
        """
        classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        classes.sort()
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx

    def load_annotations(self):
        """Load annotation file to get image(static 1 frame video)
        information."""

        video_infos = []
        if self.ann_file is not None:
            with open(self.ann_file, 'r') as fin:
                for line in fin:
                    line_split = line.strip().split()
                    filename, label = line_split
                    label = int(label)
                    if self.data_prefix is not None:
                        filename = osp.join(self.data_prefix, filename)
                    video_infos.append(
                        dict(filename=filename, label=label, total_frames=1))
        else:
            classes, class_to_idx = self._find_classes(self.data_prefix)
            samples = make_dataset(self.data_prefix, class_to_idx,
                                   IMG_EXTENSIONS, None)
            for path, class_index in samples:
                video_infos.append(
                    dict(filename=path, label=class_index, total_frames=1))

        return video_infos
