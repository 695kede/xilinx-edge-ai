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
    output_shape = net.blobs['ConvNd_91'].data.shape

    label_colours = cv2.imread(args.colours, 1).astype(np.uint8)
    input_directory = args.input_directory

for filename in os.listdir(input_directory):
    input_image = cv2.imread(os.path.join(input_directory, filename), 1).astype(np.float32)
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

    prediction = net.blobs['ConvNd_91'].data[0].argmax(axis=0)

    prediction_gray = np.squeeze(prediction)

    if args.out_dir is not None:
        input_path_ext = os.path.join(input_directory, filename).split(".")[-1]
        input_image_name = os.path.splitext(filename)[0]
        input_image_name = input_image_name.replace('_leftImg8bit', '')
        out_path_gt = args.out_dir + input_image_name + '_gtFine_labelTrainIds' + '.' + input_path_ext
        print(out_path_gt)
        cv2.imwrite(out_path_gt, prediction_gray)




