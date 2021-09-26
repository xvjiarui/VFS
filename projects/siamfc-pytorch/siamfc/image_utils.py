import numbers

import cv2
import numpy as np


def get_cropped_input(inputImage,
                      bbox,
                      padScale,
                      outputSize,
                      interpolation=cv2.INTER_LINEAR,
                      pad_color=0):
    bbox = np.array(bbox)
    width = float(bbox[2] - bbox[0])
    height = float(bbox[3] - bbox[1])
    imShape = np.array(inputImage.shape)
    if len(imShape) < 3:
        inputImage = inputImage[:, :, np.newaxis]
    xC = float(bbox[0] + bbox[2]) / 2
    yC = float(bbox[1] + bbox[3]) / 2
    boxOn = np.zeros(4)
    boxOn[0] = float(xC - padScale * width / 2)
    boxOn[1] = float(yC - padScale * height / 2)
    boxOn[2] = float(xC + padScale * width / 2)
    boxOn[3] = float(yC + padScale * height / 2)
    outputBox = boxOn.copy()
    boxOn = np.round(boxOn).astype(int)
    boxOnWH = np.array([boxOn[2] - boxOn[0], boxOn[3] - boxOn[1]])
    imagePatch = inputImage[max(boxOn[1], 0):min(boxOn[3], imShape[0]),
                            max(boxOn[0], 0):min(boxOn[2], imShape[1]), :]
    boundedBox = np.clip(boxOn, 0, imShape[[1, 0, 1, 0]])
    boundedBoxWH = np.array(
        [boundedBox[2] - boundedBox[0], boundedBox[3] - boundedBox[1]])

    if imagePatch.shape[0] == 0 or imagePatch.shape[1] == 0:
        patch = np.zeros((int(outputSize), int(outputSize), 3),
                         dtype=imagePatch.dtype)
    else:
        patch = cv2.resize(
            imagePatch,
            (
                max(1, int(
                    np.round(outputSize * boundedBoxWH[0] / boxOnWH[0]))),
                max(1, int(
                    np.round(outputSize * boundedBoxWH[1] / boxOnWH[1]))),
            ),
            interpolation=interpolation,
        )
        if len(patch.shape) < 3:
            patch = patch[:, :, np.newaxis]
        patchShape = np.array(patch.shape)

        pad = np.zeros(4, dtype=int)
        pad[:2] = np.maximum(0, -boxOn[:2] * outputSize / boxOnWH)
        pad[2:] = outputSize - (pad[:2] + patchShape[[1, 0]])

        if np.any(pad != 0):
            if len(pad[pad < 0]) > 0:
                patch = np.zeros((int(outputSize), int(outputSize), 3))
            else:
                if isinstance(pad_color, numbers.Number):
                    patch = np.pad(
                        patch, ((pad[1], pad[3]), (pad[0], pad[2]), (0, 0)),
                        'constant',
                        constant_values=pad_color)
                else:
                    patch = cv2.copyMakeBorder(
                        patch,
                        pad[1],
                        pad[3],
                        pad[0],
                        pad[2],
                        cv2.BORDER_CONSTANT,
                        value=pad_color)

    return patch, outputBox
