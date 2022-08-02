# -*- coding:utf-8 -*-
"""
@author: David Teng
"""

def bbox_IOU(GT_box, query_box):
    """
        思路：
            1. 先判断两个boxes是否有交集，没有就返回0
            2. 如果有交集，就计算IOU（面积的交并比）
        输入：
            两个box的坐标都是xyxy形式(左上和右下坐标)
        返回：
            返回的是IOU值

        这里遗留了一个问题，就是下面加/减 1的问题？？？？？
    """
    iw = min(GT_box[2], query_box[2]) - max(GT_box[0], query_box[0]) + 1  # 在x轴方向上的交集
    ih = min(GT_box[3], query_box[3]) - max(GT_box[1], query_box[1]) + 1  # 在y轴方向上的交集
    if iw <= 0 or ih <= 0:  # 没有交集或者一条边重合
        return 0
    else:                   # 有交集
        gt_area = (GT_box[3] - GT_box[1] + 1) * (GT_box[2] - GT_box[0] + 1)
        query_area = (query_box[3] - query_box[1] + 1) * (query_box[2] - query_box[0] + 1)
        sa = iw * ih    # 重合部分的面积
        ua = gt_area + query_area - sa
        return sa / ua

    pass
