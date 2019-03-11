# environmental variables: $HOME_DIR and $WORK_DIR

import os

PROJ_DIR      = os.environ['ML_DIR']                               # CIFAR10 working dir
WORK_DIR      = PROJ_DIR + "/caffe"              # CIFAR10 caffe dir

# environmental variables: $CAFFE_ROOT and $CAFFE_TOOLS_DIR
CAFFE_ROOT     = os.environ['CAFFE_ROOT']  # where your Caffe root is placed
CAFFE_TOOLS_DIR= CAFFE_ROOT  + "/distribute"             # the effective Caffe root


# project folders
PROJ_JPG_DIR    = PROJ_DIR + "/input/jpg" # where plain JPEG images are placed
INPUT_DIR       = PROJ_DIR + "/input"     # input image and databases main directory
LMDB_DIR        = INPUT_DIR + "/lmdb"                 # where validation and training LMDB databases are placed
VALID_DIR       = LMDB_DIR + "/valid_lmdb"  # i.e. "/home/danieleb/ML/cats-vs-dogs/input/lmdb/valid_lmdb"
TRAIN_DIR       = LMDB_DIR + "/train_lmdb"  # i.e. "/home/danieleb/ML/cats-vs-dogs/input/lmdb/train_lmdb"

#project file for mean values
MEAN_FILE = INPUT_DIR + "/mean.binaryproto" # i.e. "/home/danieleb/ML/cats-vs-dogs/input/cifar10_mean.binaryproto"
