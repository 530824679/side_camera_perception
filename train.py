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
import time
import numpy as np
import tensorflow as tf

from core.loss import calc_loss
from core.config import cfg
from network.detecthead import Model

class Train(object):
    def __init__(self):
        pass

    def train(self):
        # input placehold
        inputs = tf.placeholder([None, cfg.image_height, cfg.image_width, 3])
        is_train = tf.placeholder(tf.bool, shape=[])

        # build model
        model = Model(cfg.norm_epsilon, cfg.norm_decay, cfg.anchors_path, cfg.classes_path, cfg.pre_train)
        output = model.build(inputs, cfg.num_anchors / 3, cfg.num_classes, is_train)

        # loss
        loss = calc_loss(output, bbox_true, model.anchors, cfg.num_classes, cfg.ignore_thresh)
        l2_loss = tf.losses.get_regularization_loss()
        loss += l2_loss

        tf.summary.scalar('loss', loss)
        merged_summary = tf.summary.merge_all()

        # optimiter
        global_step = tf.Variable(0, trainable=False)
        lr = tf.train.exponential_decay(cfg.learning_rate, global_step, decay_steps=2000, decay_rate=0.8)
        optimizer = tf.train.AdamOptimizer(learning_rate=lr)

        # 如果读取预训练权重，则冻结darknet53网络的变量
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            if cfg.pre_train:
                train_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='yolo')
                train_op = optimizer.minimize(loss=loss, global_step=global_step, var_list=train_var)
            else:
                train_op = optimizer.minimize(loss=loss, global_step=global_step)

        init = tf.global_variables_initializer()
        saver = tf.train.Saver()
        with tf.Session(config=tf.ConfigProto(log_device_placement=False)) as sess:
            ckpt = tf.train.get_checkpoint_state(config.model_dir)
            if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
                print('restore model', ckpt.model_checkpoint_path)
                saver.restore(sess, ckpt.model_checkpoint_path)
            else:
                sess.run(init)
            if cfg.pre_train is True:
                load_ops = load_weights(tf.global_variables(scope='darknet53'), config.darknet53_weights_path)
                sess.run(load_ops)
            summary_writer = tf.summary.FileWriter(cfg.log_dir, sess.graph)

            loss_value = 0
            for epoch in range(cfg.epochs):
                for step in range(int(config.train_num / cfg.batch_size)):

                    start_time = time.time()
                    train_loss, summary, global_step_value, _ = sess.run([loss, merged_summary, global_step, train_op], {is_train: True})
                    loss_value += train_loss
                    duration = time.time() - start_time

                    examples_per_sec = float(duration) / cfg.batch_size
                    format_str = ('Epoch {} step {},  train loss = {} ( {} examples/sec; {} ''sec/batch)')
                    print(format_str.format(epoch, step, loss_value / global_step_value, examples_per_sec, duration))

                    summary_writer.add_summary(summary=tf.Summary(value=[tf.Summary.Value(tag="train loss", simple_value=train_loss)]), global_step=step)
                    summary_writer.add_summary(summary, step)
                    summary_writer.flush()

                if epoch % 5 == 0:
                    checkpoint_path = os.path.join(cfg.checkpoints_dir, 'model.ckpt')
                    saver.save(sess, checkpoint_path, global_step=global_step)
