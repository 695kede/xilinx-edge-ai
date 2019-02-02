#!/usr/bin/sh 

DNNDK_ROOT=$HOME/ML/DNNDK/tools

#working directory
work_dir=$HOME/ML/cats-vs-dogs/deephi/alexnetBNnoLRN/quantiz #$(pwd)
#path of float model
model_dir=${work_dir}
#output directory
output_dir=${work_dir}/decent_output


#soft link to the calibration data
ln -nsf $HOME/ML/cats-vs-dogs/input/jpg/calib  $HOME/ML/cats-vs-dogs/deephi/alexnetBNnoLRN/quantiz/data/calib

# copy input files from alexnetBNnoLRN via soft links
ln -nsf  $HOME/ML/cats-vs-dogs/caffe/models/alexnetBNnoLRN/m2/deephi_train_val_2_alexnetBNnoLRN.prototxt $HOME/ML/cats-vs-dogs/deephi/alexnetBNnoLRN/quantiz/float.prototxt
ln -nsf  $HOME/ML/cats-vs-dogs/caffe/models/alexnetBNnoLRN/m2/snapshot_2_alexnetBNnoLRN__iter_12000.caffemodel $HOME/ML/cats-vs-dogs/deephi/alexnetBNnoLRN/quantiz/float.caffemodel

# run DECENT
$DNNDK_ROOT/decent     quantize                   \
           -model ${model_dir}/float.prototxt     \
           -weights ${model_dir}/float.caffemodel \
           -output_dir ${output_dir} \
	   -method 1 \
	   -auto_test -test_iter 80

