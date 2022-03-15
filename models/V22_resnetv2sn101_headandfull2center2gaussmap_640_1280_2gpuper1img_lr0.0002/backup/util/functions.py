from __future__ import division
import cv2
import numpy as np
from util.nms.py_cpu_nms import py_cpu_nms
from util.nms_wrapper import nms


def resize(image, min_side=800, max_side=1400):
    rows, cols, cns = image.shape
    smallest_side = min(rows, cols)
    scale = 1.0 * min_side / smallest_side
    largest_side = max(rows, cols)

    if largest_side * scale > max_side:
        scale = 1.0 * max_side / largest_side
    image = cv2.resize(image, (int(round((cols * scale))), int(round((rows * scale)))))

    rows, cols, cns = image.shape

    pad_w = (-rows) % 32
    pad_h = (-cols) % 32

    new_image = np.zeros((rows + pad_w, cols + pad_h, cns)).astype(np.float32)
    new_image[:rows, :cols, :] = image.astype(np.float32)

    return new_image, scale


def vis_detections(im, class_det, w=None):
    for det in class_det:
        bbox = det[:4]
        score = det[4]
        cv2.rectangle(im, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (127, 255, 0), 1)
        cv2.putText(im, '{:.3f}'.format(score), (int(bbox[0]), int(bbox[1] - 9)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 0, 0), thickness=1, lineType=8)

    if w is not None:
        cv2.imwrite(w, im)


def parse_det_offset(r, poses, heights, offsets, size, score=0.1, down=4, nms_thresh=0.3):
    poses = np.squeeze(poses)
    heights = np.squeeze(heights)
    boxs = []
    for tcw in range(poses.shape[0]):
        pos = poses[tcw]
        height = heights[tcw]
        offset_y = offsets[0, 0+2*tcw, :, :]
        offset_x = offsets[0, 1+2*tcw, :, :]
        y_c, x_c = np.where(pos > score)
        if len(y_c) > 0:
            for i in range(len(y_c)):
                h = np.exp(height[y_c[i], x_c[i]]) * down
                w = r * h
                o_y = offset_y[y_c[i], x_c[i]]
                o_x = offset_x[y_c[i], x_c[i]]
                s = pos[y_c[i], x_c[i]]
                if tcw == 0:
                    x1, y1 = max(0, (x_c[i] + o_x + 0.5) * down - w / 2), max(0, (y_c[i] + o_y + 0.5) * down - h*0.1)
                elif tcw ==1:
                    x1, y1 = max(0, (x_c[i] + o_x + 0.5) * down - w / 2), max(0, (y_c[i] + o_y + 0.5) * down - h*0.5)
                else:
                    x1, y1 = max(0, (x_c[i] + o_x + 0.5) * down - w / 2), max(0, (y_c[i] + o_y + 0.5) * down - h*0.9)
                boxs.append([x1, y1, min(x1 + w, size[1]), min(y1 + h, size[0]), s, tcw])  # xyxy
    boxs = np.asarray(boxs, dtype=np.float32)
    if len(boxs)>0:
        keep = py_cpu_nms(boxs, nms_thresh)
        # keep = nms(boxs, nms_thresh, False, False)
        boxs = boxs[keep]
    return boxs
