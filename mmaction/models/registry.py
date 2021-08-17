from mmcv.utils import Registry

BACKBONES = Registry('backbone')
HEADS = Registry('head')
RECOGNIZERS = Registry('recognizer')
LOSSES = Registry('loss')
LOCALIZERS = Registry('localizer')
TRACKERS = Registry('tracker')
DROP_LAYERS = Registry('drop_layer')
