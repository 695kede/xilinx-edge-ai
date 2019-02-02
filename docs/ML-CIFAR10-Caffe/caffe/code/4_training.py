# ##################################################################################################
# USAGE
# python /home/ML/cifar10/caffe/code/4_training.py -s /home/ML/cifar10/caffe/models/miniVggNet/m3/solver_3_miniVggNet.prototxt -l /home/ML/cifar10/caffe/models/miniVggNet/m3/logfile_3_miniVggNet.log

#by daniele.bagni@xilinx.com

# ##################################################################################################



import  os
import glob
from datetime import datetime

from config import cifar10_config as config

import argparse

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-l", "--logfile", required=True, help="logfile")
ap.add_argument("-s", "--solver",  required=True, help="solver")

args = vars(ap.parse_args())
LOGFILE = args["logfile"]
SOLVER  = args["solver"]

CAFFE_TRAINING = config.CAFFE_TOOLS_DIR + "/bin/caffe.bin train "      #i.e. "/caffe/Caffe-SSD-Ristretto/distribute/bin/caffe.bin train "

# ##################################################################################################

print("TRAINING WITH CAFFE")

caffe_solver  = SOLVER 
caffe_logfile = LOGFILE

caffe_command = CAFFE_TRAINING + ' --solver ' + caffe_solver + ' 2>&1 | tee ' + caffe_logfile

startTime1 = datetime.now()
os.system(caffe_command)
endTime1 = datetime.now()
diff1 = endTime1 - startTime1
print("\n")
print("Elapsed time for Caffe training (s): ", diff1.total_seconds())
print("\n")


