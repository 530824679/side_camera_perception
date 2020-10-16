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

def residual_block(inputs, filters_num, blocks_num, conv_index, training = True, norm_decay = 0.99, norm_epsilon = 1e-3):
    # 在输入feature map的长宽维度进行padding
    inputs = tf.pad(inputs, paddings=[[0, 0], [1, 0], [1, 0], [0, 0]], mode='CONSTANT')
    layer = conv2d(inputs, filters_num, kernel_size = 3, strides = 2, name = "conv2d_" + str(conv_index))
    layer = batch_normalization(layer, name = "batch_normalization_" + str(conv_index), training = training, norm_decay = norm_decay, norm_epsilon = norm_epsilon)

    conv_index += 1
    for _ in range(blocks_num):
        shortcut = layer
        layer = conv2d(layer, filters_num // 2, kernel_size = 1, strides = 1, name = "conv2d_" + str(conv_index))
        layer = batch_normalization(layer, name = "batch_normalization_" + str(conv_index), training = training, norm_decay = norm_decay, norm_epsilon = norm_epsilon)

        conv_index += 1
        layer = conv2d(layer, filters_num, kernel_size = 3, strides = 1, name = "conv2d_" + str(conv_index))
        layer = batch_normalization(layer, name = "batch_normalization_" + str(conv_index), training = training, norm_decay = norm_decay, norm_epsilon = norm_epsilon)

        conv_index += 1
        layer += shortcut
    return layer, conv_index

def darknet53(inputs, conv_index, training=True, norm_decay=0.99, norm_epsilon=1e-3):
    # conv: return 52 layer if input shape is 416x416x3 output shape is 13x13x1024
    # route1: return 26 layer 52x52x256
    # route2: return 43 layer 26x26x512
    with tf.variable_scope('darknet53'):
        conv = conv2d(inputs, filters_num=32, kernel_size=3, strides=1, name="conv2d_" + str(conv_index))
        conv = batch_normalization(conv, name="batch_normalization_" + str(conv_index), training=training, norm_decay=norm_decay, norm_epsilon=norm_epsilon)

        conv_index += 1
        conv, conv_index = residual_block(conv, conv_index=conv_index, filters_num=64, blocks_num=1, training=training, norm_decay=norm_decay, norm_epsilon=norm_epsilon)
        conv, conv_index = residual_block(conv, conv_index=conv_index, filters_num=128, blocks_num=2, training=training, norm_decay=norm_decay, norm_epsilon=norm_epsilon)
        conv, conv_index = residual_block(conv, conv_index=conv_index, filters_num=256, blocks_num=8, training=training, norm_decay=norm_decay, norm_epsilon=norm_epsilon)

        route1 = conv
        conv, conv_index = residual_block(conv, conv_index=conv_index, filters_num=512, blocks_num=8, training=training, norm_decay=norm_decay, norm_epsilon=norm_epsilon)

        route2 = conv
        conv, conv_index = residual_block(conv, conv_index=conv_index, filters_num=1024, blocks_num=4, training=training, norm_decay=norm_decay, norm_epsilon=norm_epsilon)

    return route1, route2, conv, conv_index