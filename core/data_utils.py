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
------------
    true_boxes: ground truth box 形状为[boxes, 5], x_min, y_min, x_max, y_max, class_id
    input_shape:
    anchors:
    num_classes:
Returns
------------
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
    解析tfRecord数据
Parameters
------------
    serialized_example: 序列化的每条数据
Returns
------------


"""
def parse_serialized(serialized_example):
    features = tf.parse_single_example(
        serialized_example,
        features={
            'image/encoded': tf.FixedLenFeature([], dtype=tf.string),
            'image/object/bbox/xmin': tf.VarLenFeature(dtype=tf.float32),
            'image/object/bbox/xmax': tf.VarLenFeature(dtype=tf.float32),
            'image/object/bbox/ymin': tf.VarLenFeature(dtype=tf.float32),
            'image/object/bbox/ymax': tf.VarLenFeature(dtype=tf.float32),
            'image/object/bbox/label': tf.VarLenFeature(dtype=tf.float32)
        }
    )
    image = tf.image.decode_jpeg(features['image/encoded'], channels=3)
    image = tf.image.convert_image_dtype(image, tf.uint8)
    xmin = tf.expand_dims(features['image/object/bbox/xmin'].values, axis=0)
    ymin = tf.expand_dims(features['image/object/bbox/ymin'].values, axis=0)
    xmax = tf.expand_dims(features['image/object/bbox/xmax'].values, axis=0)
    ymax = tf.expand_dims(features['image/object/bbox/ymax'].values, axis=0)
    label = tf.expand_dims(features['image/object/bbox/label'].values, axis=0)
    bbox = tf.concat(axis=0, values=[xmin, ymin, xmax, ymax, label])
    bbox = tf.transpose(bbox, [1, 0])

    image, bbox = data_preprocess(image, bbox)
    bbox_true_13, bbox_true_26, bbox_true_52 = tf.py_func(preprocess_true_boxes, [bbox], [tf.float32, tf.float32, tf.float32])

    return image, bbox, bbox_true_13, bbox_true_26, bbox_true_52


"""
Introduction
------------
    对图片进行预处理，增强数据集
Parameters
------------
    image: tensorflow解析的图片
    bbox: 图片中对应的box坐标
"""
def data_preprocess(image, bbox):
    image_width, image_high = tf.cast(tf.shape(image)[1], tf.float32), tf.cast(tf.shape(image)[0], tf.float32)
    input_width = tf.cast(self.input_shape, tf.float32)
    input_high = tf.cast(self.input_shape, tf.float32)
    new_high = image_high * tf.minimum(input_width / image_width, input_high / image_high)
    new_width = image_width * tf.minimum(input_width / image_width, input_high / image_high)
    # 将图片按照固定长宽比进行padding缩放
    dx = (input_width - new_width) / 2
    dy = (input_high - new_high) / 2
    image = tf.image.resize_images(image, [tf.cast(new_high, tf.int32), tf.cast(new_width, tf.int32)],
                                   method=tf.image.ResizeMethod.BICUBIC)
    new_image = tf.image.pad_to_bounding_box(image, tf.cast(dy, tf.int32), tf.cast(dx, tf.int32),
                                             tf.cast(input_high, tf.int32), tf.cast(input_width, tf.int32))
    image_ones = tf.ones_like(image)
    image_ones_padded = tf.image.pad_to_bounding_box(image_ones, tf.cast(dy, tf.int32), tf.cast(dx, tf.int32),
                                                     tf.cast(input_high, tf.int32), tf.cast(input_width, tf.int32))
    image_color_padded = (1 - image_ones_padded) * 128
    image = image_color_padded + new_image
    # 矫正bbox坐标
    xmin, ymin, xmax, ymax, label = tf.split(value=bbox, num_or_size_splits=5, axis=1)
    xmin = xmin * new_width / image_width + dx
    xmax = xmax * new_width / image_width + dx
    ymin = ymin * new_high / image_high + dy
    ymax = ymax * new_high / image_high + dy
    bbox = tf.concat([xmin, ymin, xmax, ymax, label], 1)
    if self.mode == 'train':
        # 随机左右翻转图片
        def _flip_left_right_boxes(boxes):
            xmin, ymin, xmax, ymax, label = tf.split(value=boxes, num_or_size_splits=5, axis=1)
            flipped_xmin = tf.subtract(input_width, xmax)
            flipped_xmax = tf.subtract(input_width, xmin)
            flipped_boxes = tf.concat([flipped_xmin, ymin, flipped_xmax, ymax, label], 1)
            return flipped_boxes

        flip_left_right = tf.greater(tf.random_uniform([], dtype=tf.float32, minval=0, maxval=1), 0.5)
        image = tf.cond(flip_left_right, lambda: tf.image.flip_left_right(image), lambda: image)
        bbox = tf.cond(flip_left_right, lambda: _flip_left_right_boxes(bbox), lambda: bbox)
    # 将图片归一化到0和1之间
    image = image / 255.
    image = tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)
    bbox = tf.clip_by_value(bbox, clip_value_min=0, clip_value_max=input_width - 1)
    bbox = tf.cond(tf.greater(tf.shape(bbox)[0], config.max_boxes), lambda: bbox[:config.max_boxes],
                   lambda: tf.pad(bbox, paddings=[[0, config.max_boxes - tf.shape(bbox)[0]], [0, 0]], mode='CONSTANT'))
    return image, bbox












"""
Introduction
------------
    建立数据集dataset
Parameters
------------
    batch_size: batch大小
Returns
------------
    dataset: 返回tensorflow的dataset
"""
def build_dataset(batch_size):
    dataset = tf.data.TFRecordDataset(filenames=self.TfrecordFile)
    dataset = dataset.map(parse_serialized, num_parallel_calls=10)
    if self.mode == 'train':
        dataset = dataset.repeat().shuffle(9000).batch(batch_size).prefetch(batch_size)
    else:
        dataset = dataset.repeat().batch(batch_size).prefetch(batch_size)
    return dataset