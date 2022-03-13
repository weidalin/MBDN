# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

import numpy as np
import torch

def py_cpu_nms(dets, thresh_base):
    """Pure Python NMS baseline."""
    # print("dets.shape: ", dets.shape)
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
        thresh = np.where(type_channle_weight_diff, 1.0, 0.75)*thresh_base
        inds = np.where(ovr <= thresh)[0]#iou小于iou阈值的保留下来（过滤掉iou超过阈值的框）,下标从0开始

        order = order[inds + 1]
        tcw = tcw[inds + 1]
        # print(order)

    return keep

def box_area(boxes):
    """
    Computes the area of a set of bounding boxes, which are specified by its
    (x1, y1, x2, y2) coordinates.
    Arguments:
        boxes (Tensor[N, 4]): boxes for which the area will be computed. They
            are expected to be in (x1, y1, x2, y2) format
    Returns:
        area (Tensor[N]): area for each box
    """
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])


def box_iou(boxes1, boxes2):
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        boxes1 (Tensor[N, 4])
        boxes2 (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise IoU values for every element in boxes1 and boxes2
    """
    area1 = box_area(boxes1)  # 每个框的面积 (N,)
    area2 = box_area(boxes2)  # (M,)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2] # N中一个和M个比较； 所以由N，M 个
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]  # 删除面积小于0 不相交的  clamp 钳；夹钳；
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]  # 切片的用法 相乘维度减1
    iou = inter / (area1[:, None] + area2 - inter)
    return iou  # NxM， boxes1中每个框和boxes2中每个框的IoU值；


def soft_nms(boxes, scores, soft_threshold=0.01,  iou_threshold=0.7, weight_method=2, sigma=0.5):
    """
    :param boxes: [N, 4]， 此处传进来的框，是经过筛选（选取的得分TopK）之后的
    :param scores: [N]
    :param iou_threshold: 0.7
    :param soft_threshold soft nms 过滤掉得分太低的框 （手动设置）
    :param weight_method 权重方法 1. 线性 2. 高斯
    :return:
    """
    keep = []
     # 值从小到大的 索引， 索引对应的 是 元boxs索引 scores索引
    idxs = scores.argsort()
    while idxs.numel() > 0:  # 循环直到null； numel()： 数组元素个数
        # 得分最大框对应的索引, 以及对应的坐标
        # 由于scores得分会改变，所以每次都要重新排序，获取得分最大值
        idxs = scores.argsort()  # 评分排序

        if idxs.size(0) == 1:  # 就剩余一个框了；
            keep.append(idxs[-1])  # 位置不能边
            break
        keep_len = len(keep)
        max_score_index = idxs[-(keep_len + 1)]
        max_score_box = boxes[max_score_index][None, :]  # [1, 4]
        idxs = idxs[:-(keep_len + 1)]
        other_boxes = boxes[idxs]  # [?, 4]
        keep.append(max_score_index)  # 位置不能边
        ious = box_iou(max_score_box, other_boxes)  # 一个框和其余框比较 1XM
        # Soft NMS 处理， 和 得分最大框 IOU大于阈值的框， 进行得分抑制
        if weight_method == 1:   # 线性抑制  # 整个过程 只修改分数
            ge_threshod_bool = ious[0] >= iou_threshold
            ge_threshod_idxs = idxs[ge_threshod_bool]
            scores[ge_threshod_idxs] *= (1. - ious[0][ge_threshod_bool])  # 小于IoU阈值的不变
            # idxs = idxs[scores[idxs] >= soft_threshold]  # 小于soft_threshold删除， 经过抑制后 阈值会越来越小；
        elif weight_method == 2:  # 高斯抑制， 不管大不大于阈值，都计算权重
            scores[idxs] *= torch.exp(-(ious[0] * ious[0]) / sigma) # 权重(0, 1]
            # idxs = idxs[scores[idxs] >= soft_threshold]
        # else:  # NMS
        #     idxs = idxs[ious[0] <= iou_threshold]

    # keep = scores[scores > soft_threshold].int()
    keep = idxs.new(keep)  # Tensor
    keep = keep[scores[keep] > soft_threshold]  # 最后处理阈值
    return keep
    # print(keep)
    # boxes = boxes[keep]  # 保留下来的框
    # scores = scores[keep]  # soft nms抑制后得分
    # return boxes, scores
# https://github.com/DongPoLI/NMS_SoftNMS/blob/main/soft_nms.py


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
        order = order[out_inds]
        dets = dets[out_inds]

    return max_ids, np.array(weighted_boxes)