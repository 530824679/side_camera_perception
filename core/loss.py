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

import tensorflow as tf


def box_iou(self, box1, box2):
    """
    Introduction
    ------------
        计算box tensor之间的iou
    Parameters
    ----------
        box1: shape=[grid_size, grid_size, anchors, xywh]
        box2: shape=[box_num, xywh]
    Returns
    -------
        iou:
    """
    box1 = tf.expand_dims(box1, -2)
    box1_xy = box1[..., :2]
    box1_wh = box1[..., 2:4]
    box1_mins = box1_xy - box1_wh / 2.
    box1_maxs = box1_xy + box1_wh / 2.

    box2 = tf.expand_dims(box2, 0)
    box2_xy = box2[..., :2]
    box2_wh = box2[..., 2:4]
    box2_mins = box2_xy - box2_wh / 2.
    box2_maxs = box2_xy + box2_wh / 2.

    intersect_mins = tf.maximum(box1_mins, box2_mins)
    intersect_maxs = tf.minimum(box1_maxs, box2_maxs)
    intersect_wh = tf.maximum(intersect_maxs - intersect_mins, 0.)
    intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
    box1_area = box1_wh[..., 0] * box1_wh[..., 1]
    box2_area = box2_wh[..., 0] * box2_wh[..., 1]
    iou = intersect_area / (box1_area + box2_area - intersect_area)
    return iou


"""
Introduction
------------
    预测图拆分为边界框的起始点xy,宽高wh,置信度confidence,类别概率class_probs
    输出的4个值box_xy, box_wh, confidence, class_probs的范围均在0~1之间
Parameters
----------
    feats: 第I个特征图,例如shape=(?, 13, 13, 3*(5+num_classes))
    anchors: 第I层anchor box,例如[(116, 90), (156,198), (373,326)]
    num_classes: 类别个数,如3个
    input_shape: 输入图片的尺寸,例如Tensor值为(416, 416)
    is_train: 是否训练的开关
Return
----------
    box_xy:归一化的起始点xy,例如shape=(?, 13, 13, num_anchors, 2)
    box_wh:归一化的宽高wh,例如shape=(?, 13, 13, num_anchors, 2)
    box_confidence:归一化的框置信度
    box_class_probs:归一化的类别置信度
    grid:shape是(grid_size[0], grid_size[1], 1, 2)，数值为0~grid_size[0]的全遍历二元组
    predictions:分离anchors和其他数据后的feats,例如shape=(?, 13, 13, num_anchors, num_classes + 5)
"""
def pred_head(feats, anchors, num_classes, input_shape, is_train = True):
    # 统计anchors的数量
    num_anchors = len(anchors)
    # 将anchors转换为与预测图feats维度相同的Tensor,即anchors_tensor的Tensor是(1, 1, 1, 3, 2)
    anchors_tensor = tf.reshape(tf.constant(anchors, dtype=tf.float32), [1, 1, 1, num_anchors, 2])
    # 获取网格的尺寸,即预测图feats的第1~2位
    grid_size = tf.shape(feats)[1:3]
    # 创建y轴的0~grid_size[0]的组合grid_y
    grid_y = tf.tile(tf.reshape(tf.range(grid_size[0]), [-1, 1, 1, 1]), [1, grid_size[1], 1, 1])
    # 创建x轴的0~grid_size[1]的组合grid_x
    grid_x = tf.tile(tf.reshape(tf.range(grid_size[1]), [1, -1, 1, 1]), [grid_size[0], 1, 1, 1])
    # grid是遍历二元数值组合的数值，Tensor是(13, 13, 1, 2)
    grid = tf.concat([grid_x, grid_y], axis=-1)
    grid = tf.cast(grid, tf.float32)
    # 将feats的最后一维展开，将anchors与其他数据（类别数 + 4个框值 + 框置信度）分离
    predictions = tf.reshape(feats, [-1, grid_size[0], grid_size[1], num_anchors, num_classes + 5])
    # 计算起始点xy：将predictions中xy的值,经过sigmoid归一化,再加上相应的grid的二元组，再除以网格边长做归一化
    box_xy = (tf.sigmoid(predictions[..., :2]) + grid) / tf.cast(grid_size[::-1], tf.float32)
    # 计算宽高wh：将predictions中wh的值,经过exp正值化,再乘以anchors_tensor的anchor box,再除以图片宽高做归一化
    box_wh = tf.exp(predictions[..., 2:4]) * anchors_tensor / input_shape[::-1]
    # 计算框置信度box_confidence：将predictions中confidence值,经过sigmoid归一化
    box_confidence = tf.sigmoid(predictions[..., 4:5])
    # 计算类别置信度box_class_probs：将predictions中class_probs值,经过sigmoid归一化
    box_class_probs = tf.sigmoid(predictions[..., 5:])
    # 训练时计算损失is_Train设为True
    if is_train == True:
        return grid, predictions, box_xy, box_wh

    return box_xy, box_wh, box_confidence, box_class_probs


"""
Introduction
------------
    计算损失值,循环计算每一层的损失值做累加
Parameters
----------
    output: 特征图
    y_true: 
    anchors: 
    num_classes: 
    ignore_thresh: 
Return
----------
    loss:

"""
def calc_loss(output, y_true, anchors, num_classes, ignore_thresh = .5):
    loss = 0
    anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
    input_shape = [416.0, 416.0]
    grid_shapes = [tf.cast(tf.shape(output[l])[1:3], tf.float32) for l in range(3)]
    # 循环每一层计算损失值
    for index in range(len(output)):
        # 获取物体置信度object_mask,最后一个维度的第四位
        object_mask = y_true[index][..., 4:5]
        # 获取类别置信度class_probs,最后一个维度的第五位
        class_probs = y_true[index][..., 5:]
        # 预测特征图，返回全遍历二元组网格grid shape=(13, 13, 1, 2),预测值predictions shape=(?, 13, 13, 3, 5+类别数),归一化的起始点pred_xy shape=(?, 13, 13, 3, 2),归一化的宽高pred_wh shape=(?, 13, 13, 3, 2)
        grid, predictions, pred_xy, pred_wh = pred_head(output[index],
                                                        anchors[anchor_mask[index]],
                                                        num_classes,
                                                        input_shape,
                                                        is_train=True)
        # 将xy和wh组合成预测框pred_box,shape=(?, 13, 13, 3, 4)
        pred_box = tf.concat([pred_xy, pred_wh], axis=-1)
        # 获取在网格中的中心点xy,偏移数据,值的范围是0~1；y_true的第0和1位是中心点xy的相对位置,范围是0~1
        raw_true_xy = y_true[index][..., :2] * grid_shapes[index][::-1] - grid
        # 获取在网络中的wh针对于anchors的比例,再转换为log形式,范围是有正有负；y_true的第2和3位是宽高wh的相对位置，范围是0~1；
        raw_true_wh = tf.log(tf.where(tf.equal(y_true[index][..., 2:4] / anchors[anchor_mask[index]] * input_shape[::-1], 0), tf.ones_like(y_true[index][..., 2:4]), y_true[index][..., 2:4] / anchors[anchor_mask[index]] * input_shape[::-1]))
        # 计算wh权重,取值范围#2-w*h(1~2),该系数是用来调整box坐标loss的系数,
        box_loss_scale = 2 - y_true[index][..., 2:3] * y_true[index][..., 3:4]
        # 生成IoU忽略阈值掩码ignore_mask
        ignore_mask = tf.TensorArray(dtype=tf.float32, size=1, dynamic_size=True)
        object_mask_bool = tf.cast(object_mask, dtype=tf.bool)
        def loop_body(internal_index, ignore_mask):
            # true_box的shape为[box_num, 4]
            true_box = tf.boolean_mask(y_true[index][internal_index, ..., 0:4], object_mask_bool[internal_index, ..., 0])
            # 计算预测框pred_box和真值框true_box的iou
            iou = box_iou(pred_box[internal_index], true_box)
            # 计算每个true_box对应的预测的iou最大的box
            best_iou = tf.reduce_max(iou, axis=-1)
            # 抑制iou小于最大阈值anchor框
            ignore_mask = ignore_mask.write(internal_index, tf.cast(best_iou < ignore_thresh, tf.float32))
            return internal_index + 1, ignore_mask

        _, ignore_mask = tf.while_loop(lambda internal_index, ignore_mask: internal_index < tf.shape(output[0])[0], loop_body, [0, ignore_mask])
        ignore_mask = ignore_mask.stack()
        # ignore_mask的shape是(?, ?, ?, 3, 1),第0位是批次数，第1~2位是特征图尺寸
        ignore_mask = tf.expand_dims(ignore_mask, axis=-1)

        # 计算中心点xy的损失值,object_mask即是否含有物体,含有是1，不含是0
        xy_loss = object_mask * box_loss_scale * tf.nn.sigmoid_cross_entropy_with_logits(labels=raw_true_xy, logits=predictions[..., 0:2])
        # 计算宽高wh的损失值,额外乘以系数0.5,平方根降低大小物体损失不平衡问题
        wh_loss = object_mask * box_loss_scale * 0.5 * tf.square(raw_true_wh - predictions[..., 2:4])
        # 框的损失值,存在物体的损失值+不存在物体的损失值,其中乘以忽略掩码ignore_mask,忽略预测框中iou大于阈值的框。
        confidence_loss = object_mask * tf.nn.sigmoid_cross_entropy_with_logits(labels=object_mask, logits=predictions[..., 4:5]) + (1 - object_mask) * tf.nn.sigmoid_cross_entropy_with_logits(labels=object_mask, logits=predictions[..., 4:5]) * ignore_mask
        # 类别损失值
        class_loss = object_mask * tf.nn.sigmoid_cross_entropy_with_logits(labels=class_probs, logits=predictions[..., 5:])
        # 中心点xy的损失值求和取均值
        xy_loss = tf.reduce_sum(xy_loss) / tf.cast(tf.shape(output[0])[0], tf.float32)
        # 宽高wh的损失值求和取均值
        wh_loss = tf.reduce_sum(wh_loss) / tf.cast(tf.shape(output[0])[0], tf.float32)
        # 框的损失值求和取均值
        confidence_loss = tf.reduce_sum(confidence_loss) / tf.cast(tf.shape(output[0])[0], tf.float32)
        # 类别的损失值求和取均值
        class_loss = tf.reduce_sum(class_loss) / tf.cast(tf.shape(output[0])[0], tf.float32)
        # 四部分损失相加
        loss += xy_loss + wh_loss + confidence_loss + class_loss

    return loss