# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

#from util.nms.gpu_nms import gpu_nms
from util.nms.cpu_nms import cpu_nms
import numpy as np

def soft_nms(dets, sigma=0.5, Nt=0.3, threshold=0.001, method=1):

    keep = soft_nms(np.ascontiguousarray(dets, dtype=np.float32),
                        np.float32(sigma), np.float32(Nt),
                        np.float32(threshold),
                        np.uint8(method))
    return keep

def nms(dets, thresh, usegpu, gpu_id):
    """Dispatch to either CPU or GPU NMS implementations."""

    if dets.shape[0] == 0:
        return []
    if usegpu:
        return cpu_nms(dets, thresh)#gpu_nms(dets, thresh, device_id=gpu_id)
    else:
        return cpu_nms(dets, thresh)

def soft_nms(boxes, sigma=0.5, Nt=0.1, threshold=0.001, method=1):
    N = boxes.shape[0]
    pos = 0
    maxscore = 0
    maxpos = 0

# boxes = np.array([[100, 100, 150, 168, 0.63], [166, 70, 312, 190, 0.55], [
                #  221, 250, 389, 500, 0.79], [12, 190, 300, 399, 0.9], [28, 130, 134, 302, 0.3]])
    for i in range(N):
        maxscore = boxes[i, 4]
        maxpos = i

        tx1 = boxes[i, 0]
        ty1 = boxes[i, 1]
        tx2 = boxes[i, 2]
        ty2 = boxes[i, 3]
        ts = boxes[i, 4]

        pos = i + 1
    # get max box
        while pos < N:
            if maxscore < boxes[pos, 4]:
                maxscore = boxes[pos, 4]
                maxpos = pos
            pos = pos + 1

    # add max box as a detection
        boxes[i, 0] = boxes[maxpos, 0]
        boxes[i, 1] = boxes[maxpos, 1]
        boxes[i, 2] = boxes[maxpos, 2]
        boxes[i, 3] = boxes[maxpos, 3]
        boxes[i, 4] = boxes[maxpos, 4]

    # swap ith box with position of max box
        boxes[maxpos, 0] = tx1
        boxes[maxpos, 1] = ty1
        boxes[maxpos, 2] = tx2
        boxes[maxpos, 3] = ty2
        boxes[maxpos, 4] = ts

        tx1 = boxes[i, 0]
        ty1 = boxes[i, 1]
        tx2 = boxes[i, 2]
        ty2 = boxes[i, 3]
        ts = boxes[i, 4]

        pos = i + 1
    # NMS iterations, note that N changes if detection boxes fall below threshold
        while pos < N:
            x1 = boxes[pos, 0]
            y1 = boxes[pos, 1]
            x2 = boxes[pos, 2]
            y2 = boxes[pos, 3]
            s = boxes[pos, 4]

            area = (x2 - x1 + 1) * (y2 - y1 + 1)
            iw = (min(tx2, x2) - max(tx1, x1) + 1)
            if iw > 0:
                ih = (min(ty2, y2) - max(ty1, y1) + 1)
                if ih > 0:
                    ua = float((tx2 - tx1 + 1) *
                               (ty2 - ty1 + 1) + area - iw * ih)
                    ov = iw * ih / ua  # iou between max box and detection box

                    if method == 1:  # linear
                        if ov > Nt:
                            weight = 1 - ov
                        else:
                            weight = 1
                    elif method == 2:  # gaussian
                        weight = np.exp(-(ov * ov)/sigma)
                    else:  # original NMS
                        if ov > Nt:
                            weight = 0
                        else:
                            weight = 1

                    boxes[pos, 4] = weight*boxes[pos, 4]
                    # print(boxes[:, 4])

            # if box score falls below threshold, discard the box by swapping with last box
            # update N
                    if boxes[pos, 4] < threshold:
                        boxes[pos, 0] = boxes[N-1, 0]
                        boxes[pos, 1] = boxes[N-1, 1]
                        boxes[pos, 2] = boxes[N-1, 2]
                        boxes[pos, 3] = boxes[N-1, 3]
                        boxes[pos, 4] = boxes[N-1, 4]
                        N = N - 1
                        pos = pos - 1

            pos = pos + 1
    keep = [i for i in range(N)]
    return keep

def weighted_nms_dict(input_dict, thresh=0.5):
    """
    Takes a Tensorflow Object Detection API style predition dict
    and outputs dict with weighted NMS-ed bounding boxes
    """
    dets = np.hstack((input_dict['detection_boxes'],
                      np.expand_dims(input_dict['detection_scores'], 1)))
    dets = dets[:input_dict['num_detections'], :]

    max_ids, weighted_boxes = weighted_nms(dets, thresh)

    output_dict = {
                   'num_detections': len(max_ids),
                   'detection_boxes': weighted_boxes,
                   'detection_scores': input_dict['detection_scores'][max_ids],
                   'detection_classes': input_dict['detection_classes'][max_ids]
                   }
    return output_dict


def weighted_nms(dets, thresh=0.5):
    """
    Takes bounding boxes and scores and a threshold and applies
    weighted non-maximal suppression.
    """
    scores = dets[:, 4]
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]

    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]

    max_ids = []
    weighted_boxes = []
    while order.size > 0:
        i = order[0]
        max_ids.append(i)
        xx1 = np.maximum(x1[i], x1[order[:]])
        yy1 = np.maximum(y1[i], y1[order[:]])
        xx2 = np.minimum(x2[i], x2[order[:]])
        yy2 = np.minimum(y2[i], y2[order[:]])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[:]] - inter)

        in_inds = np.where(iou >= thresh)[0]
        in_dets = dets[in_inds, :]

        weights = in_dets[:, 4] * iou[in_inds]
        wbox = np.sum((in_dets[:, :4] * weights[..., np.newaxis]), axis=0) \
            / np.sum(weights)
        weighted_boxes.append(wbox)

        out_inds = np.where(iou < thresh)[0]
        # return keep
        order = order[out_inds]
        dets = dets[out_inds]
    return max_ids
    # return max_ids, np.array(weighted_boxes)