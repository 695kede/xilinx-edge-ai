#!/usr/bin/sh

DNNDK_ROOT=$HOME/ML/DNNDK/tools

#working directory
work_dir=$HOME/ML/cifar10/deephi/miniVggNet/quantiz #$(pwd)
#path of float model
model_dir=${work_dir}
#output directory
output_dir=${work_dir}/decent_output

#force a soft link to the calibration data
ln -nsf $HOME/ML/cifar10/input/cifar10_jpg/calib  $HOME/ML/cifar10/deephi/miniVggNet/quantiz/data/calib

# copy input files from miniVggNet Caffe project via soft links force (nf) 
ln -nsf $HOME/ML/cifar10/caffe/models/miniVggNet/m3/deephi_train_val_3_miniVggNet.prototxt $HOME/ML/cifar10/deephi/miniVggNet/quantiz/float.prototxt

ln -nsf $HOME/ML/cifar10/caffe/models/miniVggNet/m3/snapshot_3_miniVggNet__iter_40000.caffemodel $HOME/ML/cifar10/deephi/miniVggNet/quantiz/float.caffemodel


# run DECENT
$DNNDK_ROOT/decent     quantize                                    \
           -model ${model_dir}/float.prototxt     \
           -weights ${model_dir}/float.caffemodel \
           -output_dir ${output_dir} \
	   -method 1 \
	   -auto_test -test_iter 50

