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
import cv2
import time
import numpy as np
from PIL import Image
import tensorflow as tf
from core.config import cfg

def predict(test_path):
    IMAGE_H, IMAGE_W = 640, 416

    classes = cfg.CLASSES
    num_classes = len(classes)
    input_tensor, output_tensors = utils.read_pb_return_tensors(tf.get_default_graph(), "./checkpoint/yolov3_cpu_nms.pb", ["Placeholder:0", "concat_9:0", "mul_6:0"])

    with tf.Session() as sess:
        file_list = os.listdir(test_path)
        for file in file_list:
            image_path = os.path.join(test_path, file)
            frame = cv2.imread(image_path)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame)

            img_resized = np.array(image.resize(size=(IMAGE_H, IMAGE_W)), dtype=np.float32)
            img_resized = img_resized / 255.
            prev_time = time.time()

            boxes, scores = sess.run(output_tensors, feed_dict={input_tensor: np.expand_dims(img_resized, axis=0)})
            boxes, scores, labels = utils.cpu_nms(boxes, scores, num_classes, score_thresh=0.4, iou_thresh=0.5)
            image = utils.draw_boxes(image, boxes, scores, labels, classes, (IMAGE_H, IMAGE_W), show=False)

            curr_time = time.time()
            exec_time = curr_time - prev_time
            result = np.asarray(image)
            info = "time: %.2f ms" % (1000 * exec_time)
            cv2.putText(result, text=info, org=(50, 70), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 0, 0), thickness=2)
            cv2.namedWindow("result", cv2.WINDOW_AUTOSIZE)
            result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
            cv2.imshow("result", result)
            if cv2.waitKey(1) & 0xFF == ord('q'): break

if __name__ == '__main__':
    test_path = ""
    predict(test_path)