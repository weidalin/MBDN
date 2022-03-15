# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

import numpy as np

def py_cpu_nms(dets, thresh_base):
    """Pure Python NMS baseline."""
    print("dets.shape: ", dets.shape)
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]
    tcw = dets[:, 5]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1] #根据得分排序的序号, 从小到大排序后，再倒序，也就是从大到小。。。
    tcw = tcw[order]
    keep = []
    while order.size > 0:
        i = order[0] #得到分数最大的那个i
        type_channle_weight_diff = tcw[0] == tcw[1:]
        # print("order.shape: %s, type_channle_weight_diff.shape: %s"%(order.shape, type_channle_weight_diff.shape) )
        keep.append(i) #保留i
        #xx1, yy1, xx2, yy2分别就是并集的坐标
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h #并集的面积
        ovr = inter / (areas[i] + areas[order[1:]] - inter) #iou # 104个

        #同一个channel,执行正常nms,不同channel可能会产生冗余的框，这些框需要更低的iou阈值
        # base_thresh = np.ones(type_channle_weight_diff.shape[0])*
        thresh = np.where(type_channle_weight_diff, 1.0, 0.95)*thresh_base
        inds = np.where(ovr <= thresh)[0]#iou小于iou阈值的保留下来（过滤掉iou超过阈值的框）,下标从0开始

        order = order[inds + 1]
        tcw = tcw[inds + 1]
        # print(order)

    return keep
