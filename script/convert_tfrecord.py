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
import sys
import argparse
import numpy as np
import tensorflow as tf

"""
Introduction
------------
    将图片和boxes数据存储为tfrecord
Parameters
----------
    files:数据文件存储路径
    tfrecords: tfrecord文件存储路径
"""
def convert_tfrecord(files, tfrecords):
    image_list = []
    dataset = {}
    with open(files, 'r') as f:
        for line in f.readlines():
            boxes_list = []
            line = line.rstrip("\n")
            example = line.split(' ')
            image_path = example[0]
            image_list.append(image_path)
            boxes_num = len(example[1:])
            for i in range(boxes_num):
                value = example[i + 1].split(',')
                value = list(map(int, value))
                boxes_list.append(value)
            dataset[image_path] = boxes_list

    images_num = len(image_list)
    print(">> Processing %d images" % images_num)

    with tf.python_io.TFRecordWriter(tfrecords) as record_writer:
        for i in range(images_num):
            with tf.gfile.FastGFile(image_list[i], 'rb') as file:
                image = file.read()  # 读取除二进制文件
                xmin, xmax, ymin, ymax, label = [], [], [], [], []
                boxes = dataset[image_list[i]]  # 得到图片的boxes
                for box in boxes:
                    xmin.append(box[0])
                    ymin.append(box[1])
                    xmax.append(box[2])
                    ymax.append(box[3])
                    label.append(box[4])
                example = tf.train.Example(features=tf.train.Features(
                    feature={
                        'image/encoded' : tf.train.Feature(bytes_list = tf.train.BytesList(value = [image])),
                        'image/object/bbox/xmin' : tf.train.Feature(float_list = tf.train.FloatList(value = xmin)),
                        'image/object/bbox/xmax': tf.train.Feature(float_list = tf.train.FloatList(value = xmax)),
                        'image/object/bbox/ymin': tf.train.Feature(float_list = tf.train.FloatList(value = ymin)),
                        'image/object/bbox/ymax': tf.train.Feature(float_list = tf.train.FloatList(value = ymax)),
                        'image/object/bbox/label': tf.train.Feature(float_list = tf.train.FloatList(value = label)),
                    }
                ))
                sys.stdout.write("\r>> %d / %d" % (i + 1, images_num))
                sys.stdout.flush()
                record_writer.write(example.SerializeToString())
        print(">> Saving %d images in %s" % (images_num, tfrecords))


def main(_):
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_txt", default='../data/ImageSets/Main/dataset/train.txt')
    parser.add_argument("--val_txt", default='../data/ImageSets/Main/dataset/val.txt')
    parser.add_argument("--tfrecord_dir", default='../data/ImageSets/Main/tfrecord')
    flags = parser.parse_args()

    convert_tfrecord(flags.train_txt, os.path.join(flags.tfrecord_dir, 'train.tfrecords'))
    convert_tfrecord(flags.val_txt, os.path.join(flags.tfrecord_dir, 'test.tfrecords'))

if __name__ == "__main__":
    tf.app.run()