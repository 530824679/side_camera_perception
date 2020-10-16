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


def batch_normalization(input_layer, name=None, training=True, norm_decay=0.99, norm_epsilon=1e-3):
    bn = tf.layers.batch_normalization(inputs=input_layer,
                                             momentum=norm_decay,
                                             epsilon=norm_epsilon,
                                             center=True,
                                             scale=True,
                                             training=training,
                                             name=name)
    return tf.nn.leaky_relu(bn, alpha=0.1)

def conv2d(inputs, filters_num, kernel_size, name, use_bias=False, strides=1):
    conv = tf.layers.conv2d(
        inputs=inputs,
        filters=filters_num,
        kernel_size=kernel_size,
        strides=[strides, strides],
        kernel_initializer=tf.glorot_uniform_initializer(),
        padding=('SAME' if strides == 1 else 'VALID'),
        kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=5e-4),
        use_bias=use_bias,
        name=name)
    return conv