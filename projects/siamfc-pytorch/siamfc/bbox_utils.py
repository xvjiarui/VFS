import numpy as np

LIMIT = 99999999


# BBoxes are [x1, y1, x2, y2]
def clip_bbox(bboxes, min_clip, max_x_clip, max_y_clip):
    bboxes_out = bboxes
    added_axis = False
    if len(bboxes_out.shape) == 1:
        added_axis = True
        bboxes_out = bboxes_out[:, np.newaxis]
    bboxes_out[[0, 2]] = np.clip(bboxes_out[[0, 2]], min_clip, max_x_clip)
    bboxes_out[[1, 3]] = np.clip(bboxes_out[[1, 3]], min_clip, max_y_clip)
    if added_axis:
        bboxes_out = bboxes_out[:, 0]
    return bboxes_out


# [xMid, yMid, width, height] to [x1 y1, x2, y2]
def xywh_to_xyxy(bboxes,
                 clip_min=-LIMIT,
                 clip_width=LIMIT,
                 clip_height=LIMIT,
                 round=False):
    added_axis = False
    if isinstance(bboxes, list):
        bboxes = np.array(bboxes).astype(np.float32)
    if len(bboxes.shape) == 1:
        added_axis = True
        bboxes = bboxes[:, np.newaxis]
    bboxes_out = np.zeros(bboxes.shape)
    xMid = bboxes[0]
    yMid = bboxes[1]
    width = bboxes[2]
    height = bboxes[3]
    bboxes_out[0] = xMid - width / 2.0
    bboxes_out[1] = yMid - height / 2.0
    bboxes_out[2] = xMid + width / 2.0
    bboxes_out[3] = yMid + height / 2.0
    if clip_min != -LIMIT or clip_width != LIMIT or clip_height != LIMIT:
        bboxes_out = clip_bbox(bboxes_out, clip_min, clip_width, clip_height)
    if bboxes_out.shape[0] > 4:
        bboxes_out[4:] = bboxes[4:]
    if added_axis:
        bboxes_out = bboxes_out[:, 0]
    if round:
        bboxes_out = np.round(bboxes_out).astype(int)
    return bboxes_out
