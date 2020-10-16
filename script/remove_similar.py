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
ros_path = '/opt/ros/kinetic/lib/python2.7/dist-packages'
if ros_path in sys.path:
    sys.path.remove(ros_path)
import cv2
sys.path.append('/opt/ros/kinetic/lib/python2.7/dist-packages')

import argparse
import tensorflow as tf

def filter_image(data_path, save_path):
    file_list = os.listdir(data_path)
    file_list.sort()
    count = 0
    num = 2269
    for file in file_list:
        if count % 10 == 0:
            suffix = file.split(".")[1]
            if (suffix == 'jpg') or (suffix == 'jpeg') or (suffix == 'png'):
                frame = cv2.imread(os.path.join(data_path, file))
                frame = cv2.resize(frame, (640, 416))

                file_path = os.path.join(save_path, '%05d' % int(num) + ".jpg")
                cv2.imwrite(file_path, frame)
                num += 1
        count += 1

def rename_image(data_path, save_path):
    file_list = os.listdir(data_path)
    file_list.sort()
    num = 0
    for file in file_list:
        suffix = file.split(".")[1]
        if (suffix == 'jpg') or (suffix == 'jpeg') or (suffix == 'png'):
            frame = cv2.imread(os.path.join(data_path, file))
            file_path = os.path.join(save_path, '%05d' % int(num) + ".jpg")
            cv2.imwrite(file_path, frame)
            num += 1

def hash_compare(hash_1, hash_2):
    num = 0
    if len(hash_1) != len(hash_2):
        return -1
    for i in range(len(hash_1)):
        if hash_1[i] != hash_2[i]:
            num = num + 1
    return num

def hash_encode(image):
    hash_str = ''
    image = cv2.resize(image, (9, 8))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    for i in range(8):
        for j in range(8):
            if gray[i, j] > gray[i, j+1]:
                hash_str = hash_str + '1'
            else:
                hash_str = hash_str + '0'
    return hash_str

def main(_):
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_src_path", default='/home/chenwei/ShareDisk/N/99_Exchange/to_chenwei/09-34-29-cam_1_undistort')
    parser.add_argument("--dataset_dst_path", default='/home/chenwei/Project/side_camera_perception/data')
    args = parser.parse_args()

    #filter_image(args.dataset_src_path, args.dataset_dst_path)
    #rename_image("/home/chenwei/Project/side_camera_perception/data", "/home/chenwei/Project/side_camera_perception/data_1")

    threshold_diff = 30
    count = 0

    # record the address of the picture before and after
    image_dict = {"1": [],}

    if os.path.exists(args.dataset_src_path):
        file_list = os.listdir(args.dataset_src_path)
        file_list.sort()
        for file in file_list:
            suffix = file.split(".")[1]
            if (suffix == 'jpg') or (suffix == 'jpeg') or (suffix == 'png'):
                frame = cv2.imread(os.path.join(args.dataset_src_path, file))

                if "2" in image_dict:
                    image_dict["1"] = image_dict["2"]
                else:
                    image_dict["1"] = hash_encode(frame)
                image_dict["2"] = hash_encode(frame)
                image_diff = hash_compare(image_dict["1"], image_dict["2"])
                print("diff is %d", image_diff)
                if image_diff > threshold_diff:
                    file_path = os.path.join(args.dataset_dst_path, '%05d' % int(count) + ".jpg")
                    cv2.imwrite(file_path, frame)

                count += 1
            else:
                print("Error: file is not image type.")

    else:
        print("Error: Have no dataset directory.")


if __name__ == "__main__":
    tf.app.run()