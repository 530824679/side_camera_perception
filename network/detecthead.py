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

import os
import numpy as np
import tensorflow as tf
from network.ops import conv2d, batch_normalization
from network.backbone import darknet53

class Model(object):
    def __init__(self, norm_epsilon, norm_decay, classes_path, anchors_path, pre_train):
        self.norm_epsilon = norm_epsilon
        self.norm_decay = norm_decay
        self.classes_path = classes_path
        self.anchors_path = anchors_path
        self.pre_train = pre_train
        self.anchors = self.get_anchors()
        self.classes = self.get_classes()

    def get_classes(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    def get_anchors(self):
        anchors_path = os.path.expanduser(self.anchors_path)
        with open(anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape(-1, 2)

    def detect_block(self, inputs, filters_num, out_filters, conv_index, training=True, norm_decay=0.99, norm_epsilon=1e-3):
        conv = conv2d(inputs, filters_num=filters_num, kernel_size=1, strides=1, name="conv2d_" + str(conv_index))
        conv = batch_normalization(conv, name="batch_normalization_" + str(conv_index), training=training, norm_decay=norm_decay, norm_epsilon=norm_epsilon)
        conv_index += 1

        conv = conv2d(conv, filters_num=filters_num * 2, kernel_size=3, strides=1, name="conv2d_" + str(conv_index))
        conv = batch_normalization(conv, name="batch_normalization_" + str(conv_index), training=training, norm_decay=norm_decay, norm_epsilon=norm_epsilon)
        conv_index += 1

        conv = conv2d(conv, filters_num=filters_num, kernel_size=1, strides=1, name="conv2d_" + str(conv_index))
        conv = batch_normalization(conv, name="batch_normalization_" + str(conv_index), training=training, norm_decay=norm_decay, norm_epsilon=norm_epsilon)
        conv_index += 1

        conv = conv2d(conv, filters_num=filters_num * 2, kernel_size=3, strides=1, name="conv2d_" + str(conv_index))
        conv = batch_normalization(conv, name="batch_normalization_" + str(conv_index), training=training, norm_decay=norm_decay, norm_epsilon=norm_epsilon)
        conv_index += 1

        conv = conv2d(conv, filters_num=filters_num, kernel_size=1, strides=1, name="conv2d_" + str(conv_index))
        conv = batch_normalization(conv, name="batch_normalization_" + str(conv_index), training=training, norm_decay=norm_decay, norm_epsilon=norm_epsilon)
        conv_index += 1

        route = conv
        conv = conv2d(conv, filters_num=filters_num * 2, kernel_size=3, strides=1, name="conv2d_" + str(conv_index))
        conv = batch_normalization(conv, name="batch_normalization_" + str(conv_index), training=training, norm_decay=norm_decay, norm_epsilon=norm_epsilon)
        conv_index += 1

        conv = conv2d(conv, filters_num=out_filters, kernel_size=1, strides=1, name="conv2d_" + str(conv_index), use_bias=True)
        conv_index += 1

        return route, conv, conv_index

    def build(self, inputs, num_anchors, num_classes, training=True):

        conv_index = 1
        conv2d_26, conv2d_43, conv2d_52, conv_index = darknet53(inputs, conv_index, training=training, norm_decay=self.norm_decay, norm_epsilon=self.norm_epsilon)
        with tf.variable_scope('yolo'):
            conv2d_57, conv2d_59, conv_index = self.detect_block(conv2d_52, 512, num_anchors * (num_classes + 5), conv_index=conv_index, training=training, norm_decay=self.norm_decay, norm_epsilon=self.norm_epsilon)
            conv2d_60 = conv2d(conv2d_57, filters_num=256, kernel_size=1, strides=1, name="conv2d_" + str(conv_index))
            conv2d_60 = batch_normalization(conv2d_60, name="batch_normalization_" + str(conv_index), training=training, norm_decay=self.norm_decay, norm_epsilon=self.norm_epsilon)
            conv_index += 1

            unsample_0 = tf.image.resize_nearest_neighbor(conv2d_60, [2 * tf.shape(conv2d_60)[1], 2 * tf.shape(conv2d_60)[1]], name='upsample_0')
            route0 = tf.concat([unsample_0, conv2d_43], axis=-1, name='route_0')

            conv2d_65, conv2d_67, conv_index = self.detect_block(route0, 256, num_anchors * (num_classes + 5), conv_index=conv_index, training=training, norm_decay=self.norm_decay, norm_epsilon=self.norm_epsilon)
            conv2d_68 = conv2d(conv2d_65, filters_num=128, kernel_size=1, strides=1, name="conv2d_" + str(conv_index))
            conv2d_68 = batch_normalization(conv2d_68, name="batch_normalization_" + str(conv_index), training=training, norm_decay=self.norm_decay, norm_epsilon=self.norm_epsilon)
            conv_index += 1

            unsample_1 = tf.image.resize_nearest_neighbor(conv2d_68, [2 * tf.shape(conv2d_68)[1], 2 * tf.shape(conv2d_68)[1]], name='upsample_1')
            route1 = tf.concat([unsample_1, conv2d_26], axis=-1, name='route_1')

            _, conv2d_75, _ = self.detect_block(route1, 128, num_anchors * (num_classes + 5), conv_index=conv_index, training=training, norm_decay=self.norm_decay, norm_epsilon=self.norm_epsilon)

        return [conv2d_59, conv2d_67, conv2d_75]