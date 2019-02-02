# ##################################################################################################
# USAGE
# python 2b_compute_mean.py

# It reads the images from LMDB training database and create the mean file

# by daniele.bagni@xilinx.com

# ##################################################################################################

import os
import glob
import random
import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")
import numpy as np

from config import cats_vs_dogs_config as config

import cv2
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import caffe
from caffe.proto import caffe_pb2
import lmdb

import argparse

# ##################################################################################################
# working directories

INP_DIR  = config.INPUT_DIR                              # "/home/ML/cifar10/input"
INP_LMDB = config.LMDB_DIR + "/train_lmdb"               # "/home/ML/cifar10/input/lmdb/train_lmdb"


# ##################################################################################################
# MEAN of all training dataset images

print ('\nGenerating mean image of all training data')
mean_command =  config.CAFFE_TOOLS_DIR + "/bin/compute_image_mean.bin -backend=lmdb "


os.system(mean_command + INP_LMDB + '  ' + config.MEAN_FILE)


# ##################################################################################################
# show the mean image

blob = caffe.proto.caffe_pb2.BlobProto()
data  = open(config.MEAN_FILE).read()
blob.ParseFromString(data)

mean_array = np.asarray(blob.data, dtype=np.float32).reshape((blob.channels, blob.height, blob.width))
print " mean value channel 0: ", np.mean(mean_array[0,:,:])
print " mean value channel 1: ", np.mean(mean_array[1,:,:])
print " mean value channel 2: ", np.mean(mean_array[2,:,:])

'''
# display image of mean values
arr = np.array(caffe.io.blobproto_to_array(blob))[0, :, :, :].mean(0)
plt.imshow(arr, cmap=cm.Greys_r)
#plt.imshow(arr, cmap=cm.brg)
plt.show()
'''

# ##################################################################################################
# THE RESULT SHOULD BE SOMETHING LIKE THIS IN CASE OF HISTOGRAM EQUALIZATION:
# I0123 14:21:34.758246 84929 compute_image_mean.cpp:114] Number of channels: 3
# I0123 14:21:34.758345 84929 compute_image_mean.cpp:119] mean_value channel [0]: 106.409
# I0123 14:21:34.758442 84929 compute_image_mean.cpp:119] mean_value channel [1]: 116.049
# I0123 14:21:34.758523 84929 compute_image_mean.cpp:119] mean_value channel [2]: 124.467



