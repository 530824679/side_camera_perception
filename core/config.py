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

class Config(object):
    def __init__(self):
        self.image_height          = 416
        self.image_width           = 416
        self.batch_size            = 8
        self.epochs                = 1000
        self.data_augument         = True
        self.learn_rate            = 1e-4
        self.decay_steps           = 100
        self.decay_rate            = 0.9

        self.ignore_thresh         = .5
        self.num_anchors           = 9
        self.num_classes           = 2
        self.norm_decay            = 0.99
        self.norm_epsilon          = 1e-3
        self.pre_train             = True
        self.checkpoints_dir       = "./checkpoints"
        self.logs_dir              = "./logs"
        self.classes_path          = "./dataset/ImageSets/classes.txt"
        self.anchors_path          = "./dataset/ImageSets/anchors.txt"

cfg = Config()