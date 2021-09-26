# default cfg
default_cfg = {
    # basic parameters
    # 1e-5 for r50
    'out_scale': 0.001,
    'exemplar_sz': 120,
    'instance_sz': 255,
    'context': 0.5,
    # inference parameters
    'scale_num': 3,
    'scale_step': 1.0375,
    'scale_lr': 0.59,
    'scale_penalty': 0.9745,
    'window_influence': 0.176,
    'response_sz': 17,
    'response_up': 16,
    'total_stride': 8,
    # train parameters
    'epoch_num': 50,
    'batch_size': 8,
    'num_workers': 8,
    'initial_lr': 1e-3,
    'ultimate_lr': 1e-5,
    'weight_decay': 5e-4,
    'momentum': 0.9,
    'r_pos': 16,
    'r_neg': 0,
    'pairs_per_seq': 1,
    # customize
    'optimizer': 'Adam',
    'loss': 'focal',
    'lr_schedule': 'exp',
    'lr_step_size': 10,
    'extra_conv': True,
    'out_channels': 512,
    'reduction': 1,
    'auto_resume': True,
    'force_wd': False,
    # MMAction
    'model': {
        'backbone': {
            'frozen_stages': 4,
            'dilations': (1, 1, 2, 4),
            'strides': (1, 2, 1, 1),
            'out_indices': (3, ),
            'with_cp': False,
            'norm_eval': True
        }
    },
    'out_block_index': None,
}
