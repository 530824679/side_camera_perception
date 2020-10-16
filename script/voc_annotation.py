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
import argparse
import numpy as np
from random import shuffle
import xml.etree.ElementTree as ET
from core.config import cfg


def convert_voc_annotation(root_path, data_type, data_path, use_difficult_bbox=True):
    image_dir = os.path.join(root_path, 'JPEGImages')
    anno_dir = os.path.join(root_path, 'Annotations')
    if len(os.listdir(image_dir)) != len(os.listdir(anno_dir)):
        raise("Error:image num is not equal anno num!")

    file_list = os.listdir(image_dir)
    file_list.sort()

    shuffle(file_list)

    # divide train_data and val_data
    num_example = len(file_list)
    split = np.int(num_example * 0.8)
    train_list = file_list[:split]
    val_list = file_list[split:]

    for type in data_type:
        data_list = []
        if type == 'train':
            data_list = train_list
        else:
            data_list = val_list

        img_inds_file = os.path.join(data_path, type + '.txt')
        with open(img_inds_file, 'a') as f:
            for file in data_list:
                file_name = file.split(".")[0]
                image_path = os.path.join(root_path, 'JPEGImages', file_name + '.jpg')
                annotation = image_path
                label_path = os.path.join(root_path, 'Annotations', file_name + '.xml')
                root = ET.parse(label_path).getroot()
                objects = root.findall('object')
                for obj in objects:
                    difficult = obj.find('difficult').text.strip()
                    if (not use_difficult_bbox) and (int(difficult) == 1):
                        continue
                    bbox = obj.find('bndbox')
                    class_ind = cfg.CLASSES.index(obj.find('name').text.lower().strip())
                    xmin = bbox.find('xmin').text.strip()
                    xmax = bbox.find('xmax').text.strip()
                    ymin = bbox.find('ymin').text.strip()
                    ymax = bbox.find('ymax').text.strip()
                    annotation += ' ' + ','.join([xmin, ymin, xmax, ymax, str(class_ind)])
                print(annotation)
                f.write(annotation + "\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_path", default="../dataset")
    parser.add_argument("--data_path", default="../dataset/ImageSets")
    parser.add_argument("--data_type", default=['train', 'val'])
    flags = parser.parse_args()

    if os.path.exists(os.path.join(flags.data_path, 'train.txt')): os.remove(os.path.join(flags.data_path, 'train.txt'))
    if os.path.exists(os.path.join(flags.data_path, 'val.txt')): os.remove(os.path.join(flags.data_path, 'val.txt'))

    convert_voc_annotation(flags.root_path, flags.data_type, flags.data_path, False)