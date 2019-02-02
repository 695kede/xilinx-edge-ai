# environmental variables: $HOME_DIR and $WORK_DIR

import os

HOME_DIR = os.path.expanduser("~")


PROJ_DIR       = "/ML/cats-vs-dogs/"                          # CATSvsDOGS working dir
WORK_DIR       = HOME_DIR + PROJ_DIR + "caffe"              # CATSvsDOGS caffe dir

# environmental variables: $CAFFE_ROOT and $CAFFE_TOOLS_DIR
CAFFE_ROOT     = HOME_DIR    + "/caffe_tools/BVLC1v0-Caffe"  # where your Caffe root is placed
CAFFE_TOOLS_DIR= CAFFE_ROOT  + "/distribute"             # the effective Caffe root


# project folders
PROJ_JPG_DIR    = HOME_DIR  + PROJ_DIR + "input/jpg" # where plain JPEG images are placed
INPUT_DIR       = HOME_DIR  + PROJ_DIR + "input"     # input image and databases main directory
LMDB_DIR        = INPUT_DIR + "/lmdb"                 # where validation and training LMDB databases are placed
VALID_DIR       = LMDB_DIR + "/valid_lmdb"  # i.e. "/home/danieleb/ML/cats-vs-dogs/input/lmdb/valid_lmdb"
TRAIN_DIR       = LMDB_DIR + "/train_lmdb"  # i.e. "/home/danieleb/ML/cats-vs-dogs/input/lmdb/train_lmdb"

#project file for mean values
MEAN_FILE = INPUT_DIR + "/mean.binaryproto" # i.e. "/home/danieleb/ML/cats-vs-dogs/input/cifar10_mean.binaryproto"
