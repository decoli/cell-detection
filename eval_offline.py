from __future__ import print_function

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
from ssd import build_ssd


def test_net(net, img, each_img_path, args, cur):

    if not args.debug:
        num_images = 1
        # all detections are collected into:
        #    all_boxes[cls][image] = N x 5 array of detections in
        #    (x1, y1, x2, y2, score)
        all_boxes = [[[] for _ in range(num_images)]
                    for _ in range(len(labelmap)+1)]


        for i in range(num_images):
            ori_img = img.copy()
            img = cv2.resize(img,(300,300))
            img = torch.from_numpy(img).permute(2, 0, 1)
            img = img.float()
            w = ori_img.shape[1]
            h = ori_img.shape[0]

            x = Variable(img.unsqueeze(0))
            if args.cuda:
                x = x.cuda()
            detections = net(x).data

            # skip j = 0, because it's the background class
            for j in range(1, detections.size(1)):
                dets = detections[0, j, :]
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

                for each_box in range(boxes.shape[0]):

                    if not conf[each_box].cpu().numpy() >= args.threshold:
                        continue

                    # get predict box
                    x_1 = int(boxes[each_box, 0].cpu().numpy() + 0.5)
                    x_2 = int(boxes[each_box, 2].cpu().numpy() + 0.5)
                    y_1 = int(boxes[each_box, 1].cpu().numpy() + 0.5)
                    y_2 = int(boxes[each_box, 3].cpu().numpy() + 0.5)
                    
                    # save database
                    cur_exe = '''
                        INSERT INTO Predicts(ImagePath, X1, X2, Y1, Y2, Confidence)
                        VALUES('{image_path}', {x_1}, {x_2}, {y_1}, {y_2}, {conf})'''.format(
                            image_path=each_image_path,
                            x_1=x_1,
                            x_2=x_2,
                            y_1=y_1,
                            y_2=y_2,
                            conf=conf[each_box],
                        )
                    cur.execute(cur_exe)

                    # save the image
                    if args.no_write_image:
                        pt_1 = (x_1, y_1)
                        pt_2 = (x_2, y_2)
                        color = (0, 255, 0)
                        cv2.rectangle(ori_img, pt_1, pt_2, color, thickness=4, lineType=None, shift=None)

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
                        image = cv2.putText(ori_img, 'class:{class_name}'.format(class_name=j), org, font, 
                                        fontScale, color, thickness, cv2.LINE_AA)
                        
                        img_output_path = '_{img_path}'.format(os.path.basename(img_path=each_img_path))
                        img_output_path = os.path.join(args.output, img_output_path)
                        cv2.imwrite(img_output_path, ori_img)
    else:
        cur_exe = '''
            INSERT INTO Predicts(ImagePath, X1, X2, Y1, Y2, Confidence)
            VALUES('Demo\\Path', 0.1, 0.2, 0.3, 0.4, 0.5)'''
        cur.execute(cur_exe)
        

if __name__ == '__main__':
    num_classes = len(labelmap) + 1 # +1 for background

    parser = argparse.ArgumentParser()
    parser.add_argument('-I', '--image_dir', default=None, required=True)
    parser.add_argument('-NC', '--cuda', default=True, action='store_false')
    parser.add_argument('-M', '--trained_model', required=True)
    parser.add_argument('-T', '--threshold', type=float, default=0.5)
    parser.add_argument('-D', '--debug', action='store_true')
    parser.add_argument('-NW', '--no_write_image', action='store_false', default=True)
    parser.add_argument('-O', '--output_dir', default='output')
    args = parser.parse_args()

    if args.image_dir:
        img_path = glob.glob(os.path.join(args.image_dir, '*.jpg'))
    db_name = 'database/test.db'
    conn = sqlite3.connect(db_name)
    cur = conn.cursor()

    if not os.path.isdir(args.output_dir):
        os.mkdir(args.output_dir)
        print('there is no folder named {output_dir}, and it has been automatically created.'.format(
            output_dir=args.output_dir,
        ))

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
        Confidence FLoat
        )'''
    cur.execute(cur_exe)

    if not args.debug:
        if torch.cuda.is_available():
            if args.cuda:
                torch.set_default_tensor_type('torch.cuda.FloatTensor')
            if not args.cuda:
                print("WARNING: It looks like you have a CUDA device, but aren't using \
                    CUDA.  Run with --cuda for optimal eval speed.")
                torch.set_default_tensor_type('torch.FloatTensor')
        else:
            torch.set_default_tensor_type('torch.FloatTensor')

        with torch.no_grad():
            net = build_ssd('test', 300, num_classes)
            print('Finished loading model!')
            if args.cuda and net:
                net = net.cuda()
                cudnn.benchmark = True

            net.load_state_dict(torch.load(args.trained_model))
            net.eval()

            for each_image_path in img_path:
                img = cv2.imread(each_image_path)
                if img is not None:
                    test_net(net, img, each_image_path, args, cur)
                else:
                    print('can not read image:\n{img_path}'.format(img_path=each_image_path))
    else:
        test_net(net=None, img=None, each_image_path=None, args=args, cur=cur)

    conn.commit()
    cur.close()
    conn.close()