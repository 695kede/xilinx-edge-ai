# ##################################################################################################
# USAGE
# python 3_read_lmdb.py /home/ML/cifar10/input/lmdb/valid_lmdb /home/ML/cifar10/input/lmdb/train_lmdb

# it reads LMDB databases just created with previous script 2a_create_lmdb.py
# just to debug it

# by daniele.bagni@xilinx.com

# ##################################################################################################

import lmdb
import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")
import numpy as np

from config import cifar10_config as config

import sys
import cv2
import caffe
from caffe.proto import caffe_pb2

import matplotlib.pyplot as plt
import matplotlib.cm as cm

valid_lmdb = config.VALID_DIR #sys.argv[1] 
train_lmdb = config.TRAIN_DIR #sys.argv[2]


# ##################################################################################################
print("Now testing VALIDATION LMDB")
lmdb_env = lmdb.open(valid_lmdb)

lmdb_txn = lmdb_env.begin()
lmdb_cursor = lmdb_txn.cursor()
datum = caffe_pb2.Datum()

count1 = 0 
for key, value in lmdb_cursor:
    #print(key, value)
    datum.ParseFromString(value)

    label = datum.label
    data = caffe.io.datum_to_array(datum)
    count1 = count1 +1
    
    '''
    if (count1 % 1000 == 0):
        #CxHxW to HxWxC in cv2
        image = np.transpose(data, (1,2,0))
        #cv2.imshow('VALID', image)
        arr = np.array(image)
        plt.imshow(arr, cmap=cm.hsv)
        plt.show()
        cv2.waitKey(1)
    '''
    

    #print('{},{}'.format(key, label))
  
# ##################################################################################################
print("Now testing TRAINING LMDB")
lmdb_env = lmdb.open(train_lmdb)

lmdb_txn = lmdb_env.begin()
lmdb_cursor = lmdb_txn.cursor()
datum = caffe_pb2.Datum()

count3 = 0 
for key, value in lmdb_cursor:
    #print(key, value)
    datum.ParseFromString(value)

    label = datum.label
    data = caffe.io.datum_to_array(datum)
    count3 = count3 +1


    #import pdb; pdb.set_trace()

    '''
    if (count3 % 1000 == 0):
        #CxHxW to HxWxC in cv2
        image = np.transpose(data, (1,2,0))
        #cv2.imshow('TRAIN', image)
        arr = np.array(image)
        plt.imshow(arr, cmap=cm.hsv)
        plt.show()
        cv2.waitKey(1)
    '''
    #print('{},{}'.format(key, label))


# ##################################################################################################

print("number of images in the VALID database = %d" % count1)
print("number of images in the TRAIN database = %d" % count3)


# ##################################################################################################
# writing an LMDB

'''

import numpy as np
import lmdb
import caffe

N = 1000

# Let's pretend this is interesting data
X = np.zeros((N, 3, 32, 32), dtype=np.uint8)
y = np.zeros(N, dtype=np.int64)

# We need to prepare the database for the size. We'll set it 10 times
# greater than what we theoretically need. There is little drawback to
# setting this too big. If you still run into problem after raising
# this, you might want to try saving fewer entries in a single
# transaction.
map_size = X.nbytes * 10

env = lmdb.open('mylmdb', map_size=map_size)

with env.begin(write=True) as txn:
    # txn is a Transaction object
    for i in range(N):
        datum = caffe.proto.caffe_pb2.Datum()
        datum.channels = X.shape[1]
        datum.height = X.shape[2]
        datum.width = X.shape[3]
        datum.data = X[i].tobytes()  # or .tostring() if numpy < 1.9
        datum.label = int(y[i])
        str_id = '{:08}'.format(i)

        # The encode is only essential in Python 3
        txn.put(str_id.encode('ascii'), datum.SerializeToString())


'''
    



