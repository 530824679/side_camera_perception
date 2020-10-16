#! /usr/bin/env python3
# coding=utf-8
#================================================================
#   Copyright (C) 2020 * Ltd. All rights reserved.
#
#   Editor      : pycharm
#   File name   : train.py
#   Author      : oscar chen
#   Created date: 2020-10-13 9:50:26
#   Description :
#
#================================================================

import numpy as np
import tensorflow as tf

"""
Introduction
------------
    对训练数据的ground truth box进行预处理
Parameters
----------
    true_boxes: ground truth box 形状为[boxes, 5], x_min, y_min, x_max, y_max, class_id
    input_shape:
    anchors:
    num_classes:
Return
    y_true[0]
    y_true[1]
    y_true[2]
"""
def preprocess_true_boxes(true_boxes, input_shape, anchors, num_classes):
    # 检测类别序号是否小于类别数，避免异常数据
    assert (true_boxes[..., 4] < num_classes).all()
    # 每一层anchor box的数量
    num_layers = len(anchors) // 3
    # 预设anchor box的掩码,第1层678，第2层345，第3层012
    anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]

    '''设置y_true的初始值'''
    # 转换true_boxes类型为array 左上和右下两个坐标值和一个类别,如[184, 299, 191, 310, 0.0], shape=(?, 20, 5)
    true_boxes = np.array(true_boxes, dtype='float32')
    # 转换input_shape类型为array 输入尺寸, 如416x416
    input_shape = np.array([input_shape, input_shape], dtype='int32')
    # 得到中心点xy
    boxes_xy = (true_boxes[..., 0:2] + true_boxes[..., 2:4]) // 2.
    # 得到长宽wh
    boxes_wh = true_boxes[..., 2:4] - true_boxes[..., 0:2]
    # 中心点xy除以边长做归一化
    true_boxes[..., 0:2] = boxes_xy / input_shape[::-1]
    # 长宽wh除以边长做归一化
    true_boxes[..., 2:4] = boxes_wh / input_shape[::-1]
    # input_shape等比例降低,如输入416尺寸,三个输出层尺寸为[[13,13], [26,26], [52,52]]
    grid_shapes = [input_shape // 32, input_shape // 16, input_shape // 8]
    # 全零矩阵列表初始化y_true：[(13,13,3,5+num_class),(26,26,3,5+num_class),(52,52,3,5+num_class)]
    y_true = [np.zeros((grid_shapes[l][0], grid_shapes[l][1], len(anchor_mask[l]), 5 + num_classes), dtype='float32') for l in range(num_layers)]

    '''设置anchors的值'''
    # 将anchors增加一维,用于计算每个图中所有box和anchor互相之间的iou shape=(9,2)转为shape=(1,9,2)
    anchors = np.expand_dims(anchors, 0)
    anchors_max = anchors / 2.
    anchors_min = -anchors_max
    # 将boxes_wh中宽w大于零的位，设为True，即含有box
    valid_mask = boxes_wh[..., 0] > 0
    # 只选择存在标注框的wh,去除全0行
    wh = boxes_wh[valid_mask]
    # 为了应用广播扩充维度,如(1,2)->(1,1,2)
    wh = np.expand_dims(wh, -2)
    # wh 的shape为[box_num, 1, 2]
    boxes_max = wh / 2.
    boxes_min = -boxes_max
    # 计算标注框box与anchor box的iou值
    intersect_min = np.maximum(boxes_min, anchors_min)
    intersect_max = np.minimum(boxes_max, anchors_max)
    intersect_wh = np.maximum(intersect_max - intersect_min, 0.)
    intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
    box_area = wh[..., 0] * wh[..., 1]
    anchor_area = anchors[..., 0] * anchors[..., 1]
    iou = intersect_area / (box_area + anchor_area - intersect_area)

    '''设置y_true的值'''
    # 选择IoU最大的anchor索引
    best_anchor = np.argmax(iou, axis=-1)
    # 将对应不同比例的负责该ground turth box的位置,置为ground truth box坐标
    # t是box的序号,n是最优anchor的序号
    for t, n in enumerate(best_anchor):
        # l是层号
        for l in range(num_layers):
            # 如果最优anchor在层l中,则设置其中的值,否则默认为0
            if n in anchor_mask[l]:
                # 将归一化的值，与框长宽相乘，恢复为具体值
                i = np.floor(true_boxes[t, 0] * grid_shapes[l][1]).astype('int32')
                j = np.floor(true_boxes[t, 1] * grid_shapes[l][0]).astype('int32')
                # k是在anchor box中的序号
                k = anchor_mask[l].index(n)
                # c是类别，true_boxes的第4位
                c = true_boxes[t, 4].astype('int32')
                # 将y_true的0-3位xy和wh放入y_true中
                y_true[l][j, i, k, 0:4] = true_boxes[t, 0:4]
                # 将y_true的第4位框的置信度设为1
                y_true[l][j, i, k, 4] = 1.
                # 将y_true的第5~n位的类别设为1
                y_true[l][j, i, k, 5 + c] = 1.

    return y_true[0], y_true[1], y_true[2]

"""
Introduction
------------
    建立数据集dataset
Parameters
----------
    batch_size: batch大小
Return
------
    dataset: 返回tensorflow的dataset
"""
def build_dataset(self, batch_size):
    dataset = tf.data.TFRecordDataset(filenames=self.TfrecordFile)
    dataset = dataset.map(self.parser, num_parallel_calls=10)
    if self.mode == 'train':
        dataset = dataset.repeat().shuffle(9000).batch(batch_size).prefetch(batch_size)
    else:
        dataset = dataset.repeat().batch(batch_size).prefetch(batch_size)
    return dataset