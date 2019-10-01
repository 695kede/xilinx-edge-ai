#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This script visualize the semantic segmentation of ENet.
"""
import os
import numpy as np
from argparse import ArgumentParser
from os.path import join
import argparse
import sys
caffe_root = '../../caffe-master'  # Change this to the absolute directory to ENet Caffe
sys.path.insert(0, caffe_root + 'python')
import caffe
sys.path.append('/usr/local/lib/python2.7/site-packages')
import cv2


__author__ = 'Timo Sämann'
__university__ = 'Aschaffenburg University of Applied Sciences'
__email__ = 'Timo.Saemann@gmx.de'
__data__ = '24th May, 2017'


def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, help='.prototxt file for inference')
    parser.add_argument('--weights', type=str, required=True, help='.caffemodel file')
    parser.add_argument('--colours', type=str, required=True, help='label colours')
    parser.add_argument('--input_directory', type=str, required=True, help='input image path')
    parser.add_argument('--out_dir', type=str, default=None, help='output directory in which the segmented images '
                                                                   'should be stored')
    parser.add_argument('--gpu', type=int, default='0', help='0: gpu mode active, else gpu mode inactive')

    return parser


if __name__ == '__main__':
    parser1 = make_parser()
    args = parser1.parse_args()
    caffe.set_mode_gpu()
    caffe.set_device(args.gpu)
    net = caffe.Net(args.model, args.weights, caffe.TEST)
    
    input_shape = net.blobs['data'].data.shape
    output_shape = net.blobs['conv_u0d-score_New'].data.shape

    label_colours = cv2.imread(args.colours, 1).astype(np.uint8)
    input_directory = args.input_directory

for filename in os.listdir(input_directory):
    input_image = cv2.imread(os.path.join(input_directory, filename), 1).astype(np.float32)
    input_image = cv2.resize(input_image, (input_shape[3], input_shape[2]),interpolation=cv2.INTER_LINEAR)
    b,g,r = cv2.split(input_image)
    h = input_image.shape[0]
    w = input_image.shape[1]
    for y in range (0, h):
        for x in range (0, w):
            r[y,x] = r[y,x] * 0.022 - 0.287
            g[y,x] = g[y,x] * 0.022 - 0.325
            b[y,x] = b[y,x] * 0.022 - 0.284

    input_image=cv2.merge((b,g,r))   
    input_image = input_image.transpose((2, 0, 1))
    input_image = np.asarray([input_image])

    out = net.forward_all(**{net.inputs[0]: input_image})

    prediction = net.blobs['conv_u0d-score_New'].data[0].argmax(axis=0)

    prediction = np.squeeze(prediction)
    #prediction_gray = prediction
    prediction_gray = np.resize(prediction, (1, input_shape[2], input_shape[3]))
    prediction_gray = prediction_gray.transpose(1, 2, 0).astype(np.uint8)
    #prediction_gray = np.zeros(prediction_gray.shape, dtype=np.uint8)
    prediction_gray = cv2.resize(prediction_gray, dsize=(2048, 1024), interpolation=cv2.INTER_NEAREST)
    #prediction = np.resize(prediction, (3, input_shape[2], input_shape[3]))
    #prediction = prediction.transpose(1, 2, 0).astype(np.uint8)

    #prediction_rgb = np.zeros(prediction.shape, dtype=np.uint8)
    #label_colours_bgr = label_colours[..., ::-1]
    #cv2.LUT(prediction, label_colours_bgr, prediction_rgb)

    #cv2.imshow("UNet-Lite", prediction_rgb)
    #key = cv2.waitKey(0)

    if args.out_dir is not None:
        input_path_ext = os.path.join(input_directory, filename).split(".")[-1]
        input_image_name = os.path.splitext(filename)[0]
        input_image_name = input_image_name.replace('_leftImg8bit', '')
        #out_path_im = args.out_dir + input_image_name + 'UNet-Lite' + '.' + input_path_ext
        out_path_gt = args.out_dir + input_image_name + '_gtFine_labelTrainIds' + '.' + input_path_ext
        #print(out_path_im)
        print(out_path_gt)
        #cv2.imwrite(out_path_im, prediction_rgb)
        #prediction_gray = cv2.resize(prediction_gray,(int(2048),int(1024)))
        cv2.imwrite(out_path_gt, prediction_gray)
        # cv2.imwrite(out_path_gt, prediction) #  label images, where each pixel has an ID that represents the class






