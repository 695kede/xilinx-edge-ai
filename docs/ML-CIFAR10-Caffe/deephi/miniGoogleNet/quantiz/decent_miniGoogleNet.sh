#!/bin/bash

ML_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && cd ../../.. && pwd )"
export ML_DIR
$(which decent &> /dev/null) || export PATH=$HOME/ML/DNNDK/tools:$PATH
#DNNDK_ROOT=$HOME/ML/DNNDK/tools

net=miniGoogleNet


#working directory
work_dir=$ML_DIR/deephi/${net}/quantiz #$(pwd)
#path of float model
model_dir=${work_dir}
#output directory
output_dir=${work_dir}/decent_output

#soft link to the calibration data
ln -sf $ML_DIR/input/cifar10_jpg/calib  $ML_DIR/deephi/${net}/quantiz/data/calib

# copy input files from ${net} Caffe project via soft links 
ln -sf $ML_DIR/caffe/models/${net}/m3/deephi_train_val_3_${net}.prototxt $ML_DIR/deephi/${net}/quantiz/float.prototxt

ln -sf $ML_DIR/caffe/models/${net}/m3/snapshot_3_${net}__iter_40000.caffemodel $ML_DIR/deephi/${net}/quantiz/float.caffemodel


# run DECENT
decent     quantize                                    \
           -model ${model_dir}/float.prototxt     \
           -weights ${model_dir}/float.caffemodel \
           -output_dir ${output_dir} \
	   -method 1 \
	   -auto_test -test_iter 50

