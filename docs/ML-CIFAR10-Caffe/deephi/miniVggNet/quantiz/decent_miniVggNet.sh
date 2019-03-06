#!/bin/bash

ML_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && cd ../../.. && pwd )"
export ML_DIR
echo $ML_DIR
$(which decent &> /dev/null) || export PATH=$HOME/ML/DNNDK/tools:$PATH
#DNNDK_ROOT=$HOME/ML/DNNDK/tools

#working directory
work_dir=$ML_DIR/deephi/miniVggNet/quantiz
#path of float model
model_dir=${work_dir}
#output directory
output_dir=${work_dir}/decent_output

#force a soft link to the calibration data
ln -nsf $ML_DIR/input/cifar10_jpg/calib  $ML_DIR/deephi/miniVggNet/quantiz/data/calib

# copy input files from miniVggNet Caffe project via soft links force (nf) 
ln -nsf $ML_DIR/caffe/models/miniVggNet/m3/deephi_train_val_3_miniVggNet.prototxt $ML_DIR/deephi/miniVggNet/quantiz/float.prototxt

ln -nsf $ML_DIR/caffe/models/miniVggNet/m3/snapshot_3_miniVggNet__iter_40000.caffemodel $ML_DIR/deephi/miniVggNet/quantiz/float.caffemodel


# run DECENT
decent     quantize                                    \
           -model ${model_dir}/float.prototxt     \
           -weights ${model_dir}/float.caffemodel \
           -output_dir ${output_dir} \
	   -method 1 \
	   -auto_test -test_iter 50

