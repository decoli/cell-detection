from __future__ import print_function

import class_name
import argparse
import base64
import gc
import glob
import json
import os
import pickle
import pprint
import socket
import sqlite3
import sys
import threading
import time
from io import BytesIO
from sqlite3.dbapi2 import Timestamp

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.utils.data as data
from flask import Flask, request
from PIL import Image
from torch.autograd import Variable

from data import VOC_CLASSES as labelmap
from data import VOC_ROOT, BaseTransform, VOCAnnotationTransform, VOCDetection
from layers.box_utils import nms
from ssd import build_ssd


def test_net(net, img, img_ori, img_path, args, cur=None):
    # predict
    img_predict = img / 255.
    img_predict = torch.Tensor(img_predict)
    img_predict = img_predict.permute(0,3,1,2)
    img_predict = img_predict.cuda()
    detections = net(img_predict).data

    if not args.only_predict:
        for i, (each_img, each_img_ori, each_img_path) in enumerate(zip(img, img_ori, img_path)):
            each_img = each_img.cpu().numpy()
            w = each_img_ori.shape[1]
            h = each_img_ori.shape[0]

            # skip j = 0, because it's the background class
            for j in range(1, detections.size(1)):
                dets = detections[i, j, :]
                mask = dets[:, 0].gt(0.).expand(5, dets.size(0)).t()
                dets = torch.masked_select(dets, mask).view(-1, 5)
                if dets.size(0) == 0:
                    continue
                boxes = dets[:, 1:]
                boxes[:, 0] *= w
                boxes[:, 2] *= w
                boxes[:, 1] *= h
                boxes[:, 3] *= h
                conf = dets[:, 0]

                if args.overlap:
                    keep, count = nms(boxes, conf, overlap=args.overlap)

                for each_box in range(boxes.shape[0]):
                    if args.overlap:
                        if not each_box in keep:
                            continue
                    if not conf[each_box].cpu().numpy() >= args.threshold:
                        continue

                    # get predict box
                    x_1 = int(boxes[each_box, 0].cpu().numpy() + 0.5)
                    x_2 = int(boxes[each_box, 2].cpu().numpy() + 0.5)
                    y_1 = int(boxes[each_box, 1].cpu().numpy() + 0.5)
                    y_2 = int(boxes[each_box, 3].cpu().numpy() + 0.5)
                    
                    # save database
                    if args.output_db:
                        cur_exe = '''INSERT INTO Predicts(ImagePath, X1, X2, Y1, Y2, Confidence, Class) VALUES('{image_path}', {x_1}, {x_2}, {y_1}, {y_2}, {conf}, '{class_name}')'''.format(
                                image_path=each_img_path,
                                x_1=x_1,
                                x_2=x_2,
                                y_1=y_1,
                                y_2=y_2,
                                conf=conf[each_box],
                                class_name=class_name.class_dic[str(j)]
                            )
                        cur.execute(cur_exe)

                    # save image
                    if args.write_image:
                        pt_1 = (x_1, y_1)
                        pt_2 = (x_2, y_2)
                        color = (0, 255, 0)
                        cv2.rectangle(each_img_ori, pt_1, pt_2, color, thickness=2, lineType=None, shift=None)

                        # font
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        # org
                        org = pt_1
                        # fontScale
                        fontScale = 1
                        # Blue color in BGR
                        color = (255, 0, 0)
                        # Line thickness of 2 px
                        thickness = 2
                        # Using cv2.putText() method
                        image = cv2.putText(each_img_ori, '{class_name}'.format(class_name=class_name.class_dic[str(j)]), org, font, 
                                        fontScale, color, thickness, cv2.LINE_AA)

            if args.write_image:
                img_output_path = '_{img_path}'.format(img_path=each_img_path)
                img_output_path = os.path.join(args.output_dir, os.path.basename(img_output_path))
                cv2.imwrite(img_output_path, each_img_ori)

if __name__ == '__main__':
    # torch.set_num_threads = 6
    # OMP_NUM_THREADS = 6
    # OMP_NUM_THREADS = 6
    # cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = False
    # torch.backends.cudnn.enabled = True

    num_classes = len(labelmap) + 1 # +1 for background

    parser = argparse.ArgumentParser()
    parser.add_argument('-I', '--image_dir', default=None, required=True)
    parser.add_argument('-C', '--cuda', default=False, action='store_true')
    parser.add_argument('-M', '--trained_model', required=True)
    parser.add_argument('-T', '--threshold', type=float, default=0.5)
    parser.add_argument('-D', '--debug', action='store_true')
    parser.add_argument('-W', '--write_image', action='store_true', default=False)
    parser.add_argument('-O', '--output_dir', default='output')
    parser.add_argument('-L', '--overlap', default=None, type=float)
    parser.add_argument('-B', '--output_db_path', default='database/cell.db')
    parser.add_argument('-DB', '--output_db', default=False, action='store_true')
    parser.add_argument('-N', '--num_predict', default=1, type=int)
    parser.add_argument('-P', '--only_predict', default=False, action='store_true')
    args = parser.parse_args()

    if args.output_db:
        conn = sqlite3.connect(args.output_db_path)
        cur = conn.cursor()

        # create table
        cur_exe = '''
            CREATE TABLE IF NOT EXISTS Predicts
            (
            Id INTEGER PRIMARY KEY AUTOINCREMENT,
            ImagePath STRING,
            X1 Float,
            X2 Float,
            Y1 Float,
            Y2 Float,
            Confidence FLoat,
            Class STRING
            )'''
        cur.execute(cur_exe)

    if not os.path.isdir(args.output_dir):
        os.mkdir(args.output_dir)
        print('there is no folder named {output_dir}, and it has been automatically created.'.format(
            output_dir=args.output_dir,
        ))

    if torch.cuda.is_available():
        if args.cuda:
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        if not args.cuda:
            print("Running on CPU.")
            torch.set_default_tensor_type('torch.FloatTensor')
    else:
        torch.set_default_tensor_type('torch.FloatTensor')

    with torch.no_grad():
        net = build_ssd('test', 300, num_classes)
        if args.cuda:
            net = net.cuda()
            cudnn.benchmark = True

        net.load_state_dict(torch.load(args.trained_model))
        net.eval()
        print('Finished loading model.')

        time_cost_all_start = time.time()
        time_load_image_start = time.time()

        img_path = glob.glob(os.path.join(args.image_dir, '*.jpg'))
        img_set = []
        img_ori_set = []
        img_path_set = []
        for each_img_path in img_path:
            each_img = cv2.imread(each_img_path)
            if each_img is not None:
                img_ori_set.append(each_img)
                each_img = cv2.resize(each_img,(300,300))
                img_set.append(each_img)
                img_path_set.append(each_img_path)

        img_set_iter = iter(img_set)
        img_ori_set_iter = iter(img_ori_set)
        img_path_set_iter = iter(img_path_set)

        time_load_image_end = time.time()
        print('time cost, loading image: {:1f}s'.format(time_load_image_end - time_load_image_start))

        time_start = time.time()
        # predict
        img_predict = []
        img_ori_predict = []
        img_path_predict = []

        last_predict = False
        while True:
            if len(img_predict) == 0:
                time_img_predict_start = time.time()

            try:
                img_predict.append(next(img_set_iter))
                img_ori_predict.append(next(img_ori_set_iter))
                img_path_predict.append(next(img_path_set_iter))
            except StopIteration:
                last_predict = True

            if (len(img_predict) == args.num_predict) or last_predict:
                img_predict = torch.Tensor(img_predict)
                
                time_img_predict_end = time.time()
                print('time cost, get images to predict: {:1f}s'.format(time_img_predict_end - time_img_predict_start))

                time_predict_start = time.time()
                if args.output_db:
                    test_net(net, img_predict, img_ori_predict, img_path_predict, args, cur)
                else:
                    test_net(net, img_predict, img_ori_predict, img_path_predict, args)
                time_predict_end = time.time()
                print('time cost, the model predict: {:1f}'.format(time_predict_end - time_predict_start))

                img_predict = []
                img_path_predict = []
            
            if last_predict:
                break

        time_end = time.time()

        time_cost_all_end = time.time()
        print('time cost prediction: {time_prediction:1f}'.format(time_prediction=time_end - time_start))
        print('time cost all: {time_cost_all:1f}'.format(time_cost_all=time_cost_all_end - time_cost_all_start))

    if args.output_db:
        conn.commit()
        cur.close()
        conn.close()
