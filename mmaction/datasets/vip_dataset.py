import copy
import os
import os.path as osp
import tempfile

import mmcv
import numpy as np
from mmcv.utils import print_log
from PIL import Image
from terminaltables import AsciiTable

from mmaction.utils import add_prefix, terminal_is_available
from .rawframe_dataset import RawframeDataset
from .registry import DATASETS


@DATASETS.register_module()
class VIPDataset(RawframeDataset):

    PALETTE = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128],
               [128, 0, 128], [0, 128, 128], [128, 128, 128], [64, 0, 0],
               [191, 0, 0], [64, 128, 0], [191, 128, 0], [64, 0, 128],
               [191, 0, 128], [64, 128, 128], [191, 128, 128], [0, 64, 0],
               [128, 64, 0], [0, 191, 0], [128, 191, 0], [0, 64, 128],
               [128, 64, 128]]
    CLASSES = [
        'background', 'hat', 'hair', 'sun-glasses', 'upper-clothes', 'dress',
        'coat', 'socks', 'pants', 'gloves', 'scarf', 'skirt', 'torso-skin',
        'face', 'right-arm', 'left-arm', 'right-leg', 'left-leg', 'right-shoe',
        'left-shoe'
    ]

    def __init__(self,
                 ann_file,
                 pipeline,
                 data_prefix=None,
                 anno_prefix=None,
                 test_mode=False,
                 split='val',
                 data_root='data/vip'):
        assert split in ['train', 'val']
        self.split = split
        self.data_root = data_root
        self.anno_prefix = anno_prefix
        super().__init__(
            ann_file,
            pipeline,
            data_prefix,
            test_mode,
            filename_tmpl='{:012}.jpg',
            with_offset=False,
            multi_class=False,
            num_classes=None,
            start_index=0,
            modality='RGB')

    def prepare_test_frames(self, idx):
        """Prepare the frames for testing given the index."""
        results = copy.deepcopy(self.video_infos[idx])
        results['filename_tmpl'] = self.filename_tmpl
        results['modality'] = self.modality
        results['start_index'] = self.start_index
        ann_frame_dir = results['frame_dir'].replace(self.data_prefix,
                                                     self.anno_prefix)
        frame_list = list(sorted(os.listdir(results['frame_dir'])))
        ann_list = list(sorted(os.listdir(ann_frame_dir)))
        results['frame_list'] = frame_list
        results['seg_map'] = osp.join(ann_frame_dir, ann_list[0])
        return self.pipeline(results)

    def vip_evaluate(self, results, output_dir, logger=None):
        eval_results = {}
        assert len(results) == len(self)
        for vid_idx in range(len(self)):
            assert len(results[vid_idx]) == \
                   self.video_infos[vid_idx]['total_frames'] or \
                   isinstance(results[vid_idx], str)
        if output_dir is None:
            tmp_dir = tempfile.TemporaryDirectory()
            output_dir = tmp_dir.name
        else:
            tmp_dir = None
            mmcv.mkdir_or_exist(output_dir)

        if terminal_is_available():
            prog_bar = mmcv.ProgressBar(len(self))
        pred_path = []
        gt_path = []
        for vid_idx in range(len(results)):
            cur_results = results[vid_idx]
            frame_dir = self.video_infos[vid_idx]['frame_dir']
            ann_frame_dir = frame_dir.replace(self.data_prefix,
                                              self.anno_prefix)
            frame_list = list(sorted(os.listdir(frame_dir)))
            ann_list = list(sorted(os.listdir(ann_frame_dir)))
            if isinstance(cur_results, str):
                file_path = cur_results
                cur_results = np.load(file_path)
                os.remove(file_path)
            for img_idx in range(self.video_infos[vid_idx]['total_frames']):
                result = cur_results[img_idx].astype(np.uint8)
                img = Image.fromarray(result)
                img.putpalette(
                    np.asarray(self.PALETTE, dtype=np.uint8).ravel())
                save_path = osp.join(
                    output_dir, osp.relpath(frame_dir, self.data_prefix),
                    frame_list[img_idx].replace('.jpg', '.png'))
                mmcv.mkdir_or_exist(osp.dirname(save_path))
                img.save(save_path)
                pred_path.append(save_path)
                gt_path.append(osp.join(ann_frame_dir, ann_list[img_idx]))
            if terminal_is_available():
                prog_bar.update()
        num_classes = len(self.CLASSES)
        class_names = self.CLASSES
        from mmaction.core.evaluation.iou import mean_iou
        ret_metrics = mean_iou(
            pred_path, gt_path, num_classes, ignore_index=255)
        ret_metrics_round = [
            np.round(ret_metric * 100, 2) for ret_metric in ret_metrics
        ]
        metric = ['mIoU']
        class_table_data = [['Class'] + [m[1:] for m in metric] + ['Acc']]
        for i in range(num_classes):
            class_table_data.append([class_names[i]] +
                                    [m[i] for m in ret_metrics_round[2:]] +
                                    [ret_metrics_round[1][i]])
        summary_table_data = [['Scope'] +
                              ['m' + head
                               for head in class_table_data[0][1:]] + ['aAcc']]
        ret_metrics_mean = [
            np.round(np.nanmean(ret_metric) * 100, 2)
            for ret_metric in ret_metrics
        ]
        summary_table_data.append(['global'] + ret_metrics_mean[2:] +
                                  [ret_metrics_mean[1]] +
                                  [ret_metrics_mean[0]])
        print_log('per class results:', logger)
        table = AsciiTable(class_table_data)
        print_log('\n' + table.table, logger=logger)
        print_log('Summary:', logger)
        table = AsciiTable(summary_table_data)
        print_log('\n' + table.table, logger=logger)

        for i in range(1, len(summary_table_data[0])):
            eval_results[summary_table_data[0]
                         [i]] = summary_table_data[1][i] / 100.0

        if tmp_dir is not None:
            tmp_dir.cleanup()

        return eval_results

    def evaluate(self, results, metrics='mIoU', output_dir=None, logger=None):
        metrics = metrics if isinstance(metrics, (list, tuple)) else [metrics]
        allowed_metrics = ['mIoU']
        for metric in metrics:
            if metric not in allowed_metrics:
                raise KeyError(f'metric {metric} is not supported')
        eval_results = dict()
        if mmcv.is_seq_of(results, np.ndarray) and results[0].ndim == 4:
            num_feats = results[0].shape[0]
            for feat_idx in range(num_feats):
                cur_results = [result[feat_idx] for result in results]
                eval_results.update(
                    add_prefix(
                        self.vip_evaluate(cur_results, output_dir, logger),
                        prefix=f'feat_{feat_idx}'))
        elif mmcv.is_seq_of(results, list):
            num_feats = len(results[0])
            for feat_idx in range(num_feats):
                cur_results = [result[feat_idx] for result in results]
                eval_results.update(
                    add_prefix(
                        self.vip_evaluate(cur_results, output_dir, logger),
                        prefix=f'feat_{feat_idx}'))
        else:
            eval_results.update(self.vip_evaluate(results, output_dir, logger))
        copypaste = []
        for k, v in eval_results.items():
            if 'mIoU' in k:
                copypaste.append(f'{float(v)*100:.2f}')
        print_log(f'Results copypaste  {",".join(copypaste)}', logger=logger)
        return eval_results
