import os


CIFAR_DIR      = os.environ['ML_DIR']                               # CIFAR10 working dir
WORK_DIR       = CIFAR_DIR + "/caffe"              # CIFAR10 caffe dir

# environmental variables: $CAFFE_ROOT and $CAFFE_TOOLS_DIR
CAFFE_ROOT     = os.environ['CAFFE_ROOT']  # where your Caffe root is placed
CAFFE_TOOLS_DIR= CAFFE_ROOT  + "/distribute"             # the effective Caffe root

# project folders
CIFAR10_JPG_DIR    = CIFAR_DIR + "/input/cifar10_jpg" # where plain CIFAR10 JPEG images are placed
INPUT_DIR          = CIFAR_DIR + "/input"             # input image and databases main directory
LMDB_DIR           = INPUT_DIR + "/lmdb"                         # where validation and training LMDB databases are placed
VALID_DIR          = LMDB_DIR + "/valid_lmdb"  # i.e. "/home/danieleb/ML/cifar10/input/lmdb/valid_lmdb"
TRAIN_DIR          = LMDB_DIR + "/train_lmdb"  # i.e. "/home/danieleb/ML/cifar10/input/lmdb/train_lmdb"

#project file for mean values
MEAN_FILE = INPUT_DIR + "/mean.binaryproto" # i.e. "/home/danieleb/ML/cifar10/input/cifar10_mean.binaryproto"
