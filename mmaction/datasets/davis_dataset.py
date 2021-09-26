import copy
import os
import os.path as osp
import tempfile

import mmcv
import numpy as np
import pandas as pd
from davis2017.evaluation import DAVISEvaluation
from mmcv.utils import print_log
from PIL import Image

from mmaction.utils import add_prefix, terminal_is_available
from .rawframe_dataset import RawframeDataset
from .registry import DATASETS


@DATASETS.register_module()
class DavisDataset(RawframeDataset):

    PALETTE = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128],
               [128, 0, 128], [0, 128, 128], [128, 128, 128], [64, 0, 0],
               [191, 0, 0], [64, 128, 0], [191, 128, 0], [64, 0, 128],
               [191, 0, 128], [64, 128, 128], [191, 128, 128], [0, 64, 0],
               [128, 64, 0], [0, 191, 0], [128, 191, 0], [0, 64, 128],
               [128, 64, 128]]

    def __init__(self,
                 ann_file,
                 pipeline,
                 data_prefix=None,
                 anno_prefix=None,
                 test_mode=False,
                 split='val',
                 data_root='data/davis2017',
                 task='semi-supervised'):
        assert split in ['train', 'val']
        assert task in ['semi-supervised']
        self.split = split
        self.data_root = data_root
        self.task = task
        self.anno_prefix = anno_prefix
        super().__init__(
            ann_file,
            pipeline,
            data_prefix,
            test_mode,
            filename_tmpl='{:05}.jpg',
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
        results['seg_map'] = osp.join(
            ann_frame_dir,
            self.filename_tmpl.format(0).replace('jpg', 'png'))
        return self.pipeline(results)

    def davis_evaluate(self, results, output_dir, logger=None):
        dataset_eval = DAVISEvaluation(
            davis_root=self.data_root, task=self.task, gt_set=self.split)
        if isinstance(results, str):
            metrics_res = dataset_eval.evaluate(results)
        else:
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
            for vid_idx in range(len(results)):
                cur_results = results[vid_idx]
                if isinstance(cur_results, str):
                    file_path = cur_results
                    cur_results = np.load(file_path)
                    os.remove(file_path)
                for img_idx in range(
                        self.video_infos[vid_idx]['total_frames']):
                    result = cur_results[img_idx].astype(np.uint8)
                    img = Image.fromarray(result)
                    img.putpalette(
                        np.asarray(self.PALETTE, dtype=np.uint8).ravel())
                    frame_dir = self.video_infos[vid_idx]['frame_dir']
                    save_path = osp.join(
                        output_dir, osp.relpath(frame_dir, self.data_prefix),
                        self.filename_tmpl.format(img_idx).replace(
                            'jpg', 'png'))
                    mmcv.mkdir_or_exist(osp.dirname(save_path))
                    img.save(save_path)
                if terminal_is_available():
                    prog_bar.update()
            metrics_res = dataset_eval.evaluate(output_dir)
            if tmp_dir is not None:
                tmp_dir.cleanup()

        J, F = metrics_res['J'], metrics_res['F']

        # Generate dataframe for the general results
        g_measures = [
            'J&F-Mean', 'J-Mean', 'J-Recall', 'J-Decay', 'F-Mean', 'F-Recall',
            'F-Decay'
        ]
        final_mean = (np.mean(J['M']) + np.mean(F['M'])) / 2.
        g_res = np.array([
            final_mean,
            np.mean(J['M']),
            np.mean(J['R']),
            np.mean(J['D']),
            np.mean(F['M']),
            np.mean(F['R']),
            np.mean(F['D'])
        ])
        g_res = np.reshape(g_res, [1, len(g_res)])
        print_log(f'\nGlobal results for {self.split}', logger=logger)
        table_g = pd.DataFrame(data=g_res, columns=g_measures)
        print_log('\n' + table_g.to_string(index=False), logger=logger)

        # Generate a dataframe for the per sequence results
        seq_names = list(J['M_per_object'].keys())
        seq_measures = ['Sequence', 'J-Mean', 'F-Mean']
        J_per_object = [J['M_per_object'][x] for x in seq_names]
        F_per_object = [F['M_per_object'][x] for x in seq_names]
        table_seq = pd.DataFrame(
            data=list(zip(seq_names, J_per_object, F_per_object)),
            columns=seq_measures)
        print_log(f'\nPer sequence results for  {self.split}', logger=logger)
        print_log('\n' + table_seq.to_string(index=False), logger=logger)

        eval_results = table_g.to_dict('records')[0]

        return eval_results

    def evaluate(self, results, metrics='daivs', output_dir=None, logger=None):
        metrics = metrics if isinstance(metrics, (list, tuple)) else [metrics]
        allowed_metrics = ['davis']
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
                        self.davis_evaluate(cur_results, output_dir, logger),
                        prefix=f'feat_{feat_idx}'))
        elif mmcv.is_seq_of(results, list):
            num_feats = len(results[0])
            for feat_idx in range(num_feats):
                cur_results = [result[feat_idx] for result in results]
                eval_results.update(
                    add_prefix(
                        self.davis_evaluate(cur_results, output_dir, logger),
                        prefix=f'feat_{feat_idx}'))
        else:
            eval_results.update(
                self.davis_evaluate(results, output_dir, logger))
        copypaste = []
        for k, v in eval_results.items():
            if 'J&F' in k:
                copypaste.append(f'{float(v)*100:.2f}')
        print_log(f'Results copypaste  {",".join(copypaste)}', logger=logger)
        return eval_results
