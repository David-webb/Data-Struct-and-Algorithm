# -*- coding: utf-8 -*-

"""
@author David Teng
"""

import numpy as np

def bboxes2deltas(gt_boxes, boxes):
    """
        计算回归系数
        输入是两个box集合, 每个集合中的任意box都是xyxy形式
    """

    # 计算boxes的宽高
    anchors_w = boxes[:, 2] - boxes[:, 0] + 1.0
    anchors_h = boxes[:, 3] - boxes[:, 1] + 1.0

    # 计算boxes的中心点
    anchors_cx = boxes[:, 0] + 0.5 * anchors_w
    anchors_cy = boxes[:, 1] + 0.5 * anchors_h
    
    # 计算gt_boxes的宽高
    gt_w = gt_boxes[:, 2] - gt_boxes[:, 0] + 1.0
    gt_h = gt_boxes[:, 3] - gt_boxes[:, 1] + 1.0

    # 计算gt_boxes的中心点
    gt_cx = gt_boxes[:, 0] + 0.5 * gt_w
    gt_cy = gt_boxes[:, 1] + 0.5 * gt_h

    # 计算回归系数
    targets_dx = (gt_cx - anchors_cx) / anchors_w
    targets_dy = (gt_cy - anchors_cy) / anchors_h
    targets_dw = np.log(gt_w / anchors_w)
    targets_dh = np.log(gt_h / anchors_h)
    deltas = np.vstack((targets_dx, targets_dy, targets_dw, targets_dh)).transpose()

    return deltas
    pass


def deltas2bboxes(bboxes, deltas):
    """
        根据回归系数修正anchors
    """
    # 提取bboxes的宽高
    widths = bboxes[:, 2] - bboxes[:, 0] + 1.0
    heights = bboxes[:, 3] - bboxes[:, 1] + 1.0

    # 提取bboxes的中心点
    b_cx = bboxes[:,0] + 0.5 * widths
    b_cy = bboxes[:,1] + 0.5 * heights

    # 提取平移回归系数
    dx = deltas[:, 0::4]
    dy = deltas[:, 1::4]

    # 提取宽高的缩放回归系数
    dw = deltas[:, 2::4]
    dh = deltas[:, 3::4]
     
    # 计算回归系数修正后的中心点和宽高
    new_cx = b_cx[:, np.newaxis] + widths[:, np.newaxis] * dx
    new_cy = b_cy[:, np.newaxis] + heights[:, np.newaxis] * dy
    new_w = np.exp(dw) * widths[:, np.newaxis]
    new_h = np.exp(dh) * heights[:, np.newaxis]

    # 计算回归后的anchor的top-left和bottom-right
    proposals = np.zeros(deltas.shape, dtype=deltas.dtype)
    proposals[:,0::4]  = new_cx - 0.5 * new_w
    proposals[:,1::4] = new_cy - 0.5 * new_h
    proposals[:,2::4]= new_cx + 0.5 * new_w
    proposals[:,0::4] = new_cy + 0.5 * new_h
     
    return proposals
    pass
