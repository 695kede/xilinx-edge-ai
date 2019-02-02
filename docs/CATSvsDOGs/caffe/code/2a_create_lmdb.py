'''
Title           :create_lmdb.py
Description     :This script divides the training images into 2 sets and stores them in lmdb databases for training and test
Author          :Adil Moujahid
Date Created    :2016-06-19
version         :0.2
usage           :python create_lmdb.py
python_version  :2.7.11
history         :modified by daniele.bagni@xilinx.com
Date Modified   :2018-04-05

modified        : daniele.bagni@xilinx.com
2018 July 07    : split test into validation (9000 img) and test (1000 img)
'''

# ##################################################################################################
# USAGE
# python 2a_create_lmdb.py -i /home/ML/cats-vs-dogs/input/jpg/ -o /home/ML/cats-vs-dogs/input/lmdb

# it reads the CIFAR10 JPG images and creates 2 LMDB databases:
# train_lmdb (20000 images in LMDB) and  val_lmdb (4000 images in LMDB) to be used during the training

# ##################################################################################################

import os
import glob
import random
import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")
import numpy as np
import cv2
import sys
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from config import cats_vs_dogs_config as config

import caffe
from caffe.proto import caffe_pb2
import lmdb

import argparse


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--inp", default = config.PROJ_JPG_DIR, help="path to the input  jpeg dir")
ap.add_argument("-o", "--out", default = config.LMDB_DIR, help="path to the output lmdb dir")
args = vars(ap.parse_args())

# this is the directory where input JPG images are placed
IMG_DIR = args["inp"]   
# this is the directory where lmdb will be placed
WORK_DIR= args["out"]   

if (not os.path.exists(WORK_DIR)): # create "WORK_DIR" directory if it does not exist
    os.mkdir(WORK_DIR)


#Size of images
IMAGE_WIDTH  = 256 #227
IMAGE_HEIGHT = 256 #227

# ##################################################################################################


def transform_img(img, img_width=IMAGE_WIDTH, img_height=IMAGE_HEIGHT):

    #Histogram Equalization
    #img[:, :, 0] = cv2.equalizeHist(img[:, :, 0])
    #img[:, :, 1] = cv2.equalizeHist(img[:, :, 1])
    #img[:, :, 2] = cv2.equalizeHist(img[:, :, 2])

    #Image Resizing
    img = cv2.resize(img, (img_width, img_height), interpolation = cv2.INTER_CUBIC)

    return img


def make_datum(img, label):
    #image is numpy.ndarray format. BGR instead of RGB
    return caffe_pb2.Datum(
        channels=3,
        width=IMAGE_WIDTH,
        height=IMAGE_HEIGHT,
        label=label,
        data=np.rollaxis(img, 2).tostring())

# ##################################################################################################

train_lmdb = WORK_DIR + '/train_lmdb'
valid_lmdb = WORK_DIR + '/valid_lmdb'

os.system('rm -rf  ' + train_lmdb)
os.system('rm -rf  ' + valid_lmdb)

if (not os.path.exists(train_lmdb)): # create directory if it does not exist
        os.mkdir(train_lmdb)
if (not os.path.exists(valid_lmdb)): # create directory if it does not exist
        os.mkdir(valid_lmdb)        

train_data = [img for img in glob.glob(IMG_DIR + "/train/*/*.jpg")]
valid_data = [img for img in glob.glob(IMG_DIR + "/val/*/*.jpg")]

# ##################################################################################################

print 'Creating train_lmdb'

#Shuffle train_data
random.seed(48)
random.shuffle(train_data)

num_train_images = 0
in_db = lmdb.open(train_lmdb, map_size=int(1e12))
with in_db.begin(write=True) as in_txn:
    for in_idx, img_path in enumerate(train_data):

        num_train_images = 1 + num_train_images
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        #img = transform_img(img, img_width=IMAGE_WIDTH, img_height=IMAGE_HEIGHT)
        filename = img_path.split("/jpg/")[1]
        if 'cat' in filename:
            label = 0
        elif 'dog' in filename:
            label = 1
        else:
            print "ERROR: there is a class which is not part of the dataset"
            sys.exit(0)
            
        datum = make_datum(img, label)
        in_txn.put('{:0>5d}'.format(in_idx), datum.SerializeToString())
        '{:0>5d}'.format(in_idx) + ':' + img_path

in_db.close()

# ##################################################################################################
# we create the validation LMDB
print '\nCreating valid_lmdb'
random.seed(48)
random.shuffle(valid_data)

num_valid_images = 0
in_db = lmdb.open(valid_lmdb, map_size=int(1e12))
with in_db.begin(write=True) as in_txn:
    for in_idx, img_path in enumerate(valid_data):

        num_valid_images = 1 + num_valid_images
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        #img = transform_img(img, img_width=IMAGE_WIDTH, img_height=IMAGE_HEIGHT)
        filename = img_path.split("/jpg/")[1]
        if 'cat' in filename:
            label = 0
        elif 'dog' in filename:
            label = 1
        else:
            print "ERROR: there is a class which is not part of the dataset"
            sys.exit(0)        

        datum = make_datum(img, label)
        in_txn.put('{:0>5d}'.format(in_idx), datum.SerializeToString()) #DB
        '{:0>5d}'.format(in_idx) + ':' + img_path
in_db.close()


print '\nFinished processing all images'
print '\n Number of images in LMDB training   dataset ',  num_train_images
print '\n Number of images in LMDB validation dataset ',  num_valid_images

