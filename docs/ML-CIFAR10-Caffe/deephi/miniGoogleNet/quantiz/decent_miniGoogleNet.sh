#!/usr/bin/sh

DNNDK_ROOT=$HOME/ML/DNNDK/tools

net=miniGoogleNet


#working directory
work_dir=$HOME/ML/cifar10/deephi/${net}/quantiz #$(pwd)
#path of float model
model_dir=${work_dir}
#output directory
output_dir=${work_dir}/decent_output

#soft link to the calibration data
ln -s $HOME/ML/cifar10/input/cifar10_jpg/calib  $HOME/ML/cifar10/deephi/${net}/quantiz/data/calib

# copy input files from ${net} Caffe project via soft links 
ln -s $HOME/ML/cifar10/caffe/models/${net}/m3/deephi_train_val_3_${net}.prototxt $HOME/ML/cifar10/deephi/${net}/quantiz/float.prototxt

ln -s $HOME/ML/cifar10/caffe/models/${net}/m3/snapshot_3_${net}__iter_40000.caffemodel $HOME/ML/cifar10/deephi/${net}/quantiz/float.caffemodel


# run DECENT
$DNNDK_ROOT/decent     quantize                                    \
           -model ${model_dir}/float.prototxt     \
           -weights ${model_dir}/float.caffemodel \
           -output_dir ${output_dir} \
	   -method 1 \
	   -auto_test -test_iter 50

