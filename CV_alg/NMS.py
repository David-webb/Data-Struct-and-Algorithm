# -*- coding:utf-8 -*- 

"""
    该代码实现了nms算法，在这里的主要作用就是对于模型预测出的验证码中的字符进行nms,同一区域只保留最大score的label
"""
import numpy as np

def captcha_nms(bboxes, scores, thresh):
    """
        bboxes: 所有的目标的bounding box
        scores: 与bboxes对应的每个对象的score
        thresh: nms使用的阈值
    """
    # 类型验证,并将bboxes和scores都转换成numpy数组
    # 计算所有bboxes的面积
    # 按照score对bboxes进行降序排序，下标保存在order列表中
    # 进行nms,将保留的对象的下标保存在keep列表中返回
        # nms算法思想：
        #   准备临时列表supressed,用于记录被抑制的box的索引
        #   接着，依次从order中选取对象，并借助supressed判断是否被抑制，如果是就跳过，选择下一个对象，否则进行如下操作：
        #       将当前对象下标添加进keep, 并计算其box面积，并依次与order中当前对象后面的对象的box计算IOU, 大于阈值的对象添加进supressed列表，表示抑制
        #   如此循环直到遍历一遍order

    assert isinstance(bboxes, np.ndarray), "传入的bboxes需要是numpy array类型"
    assert isinstance(scores, np.ndarray), "传入的scores需要是numpy array类型"
    
    #bboxes =  np.array(bboxes)
    #scores = np.array(scores)
    
    x1 = bboxes[:, 0]
    y1 = bboxes[:, 1]
    x2 = bboxes[:, 2]
    y2 = bboxes[:, 3]
    
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1] # 降序排序

    nobj = order.shape[0] # 对象总个数
    suppressed = np.zeros((nobj), dtype=np.int)
    keep = []
    for i in range(nobj):
        _i = order[i]
        if 1 == suppressed[_i]:
            continue
        keep.append(_i)
        iarea = areas[_i]   # 当前keep对象的面积
        for j in range(i+1, nobj):
            _j = order[j] 
            if 1 == suppressed[_j]:
                continue
            jarea = areas[_j]  # 待定对象
            lx_crs = max(x1[_i], x1[_j])
            ly_crs = max(y1[_i], y1[_j])
            rx_crs = min(x2[_i], x2[_j])
            ry_crs = min(y2[_i], y2[_j]) 
            w = max(0.0, (rx_crs - lx_crs + 1)) # 有可能不相交 
            h = max(0.0, (ry_crs - ly_crs + 1))
            crs_area = w * h
            iou = crs_area * 1.0 / (iarea + jarea - crs_area)
            if iou >= thresh:
                suppressed[_j] = 1
    return keep
    

def filter_cha(bboxes, labels, scores, keep):
    """
        按照keep中的索引筛选出保留的对象，并且按照bbox的位置顺序重新排序
    """
    bboxes = np.array([bboxes[i] for i in keep])
    labels = [labels[i] for i in keep]
    scores = [scores[i] for i in keep]
    x2 = bboxes[:,2]
    order = x2.argsort()
    bboxes = [bboxes[i] for i in order]
    labels = [labels[i] for i in order]
    scores = [scores[i] for i in order]
    return bboxes, labels, scores
