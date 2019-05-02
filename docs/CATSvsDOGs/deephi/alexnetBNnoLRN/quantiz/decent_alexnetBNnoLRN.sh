#!/usr/bin/bash


ML_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && cd ../../.. && pwd )"
export ML_DIR
echo $ML_DIR
$(which decent &> /dev/null) || export PATH=$HOME/ML/DNNDK/tools:$PATH
#DNNDK_ROOT=$HOME/ML/DNNDK/tools

#working directory
work_dir=$ML_DIR/deephi/alexnetBNnoLRN/quantiz #$(pwd)
#path of float model
model_dir=${work_dir}
#output directory
output_dir=${work_dir}/decent_output


#soft link to the calibration data
ln -nsf $ML_DIR/input/jpg/calib  $ML_DIR/deephi/alexnetBNnoLRN/quantiz/data/calib

# copy input files from alexnetBNnoLRN via soft links
ln -nsf  $ML_DIR/caffe/models/alexnetBNnoLRN/m2/deephi_train_val_2_alexnetBNnoLRN.prototxt $ML_DIR/deephi/alexnetBNnoLRN/quantiz/float.prototxt
ln -nsf  $ML_DIR/caffe/models/alexnetBNnoLRN/m2/snapshot_2_alexnetBNnoLRN__iter_12000.caffemodel $ML_DIR/deephi/alexnetBNnoLRN/quantiz/float.caffemodel

# run DECENT
decent     quantize                   \
           -model ${model_dir}/float.prototxt     \
           -weights ${model_dir}/float.caffemodel \
           -output_dir ${output_dir} \
	   -method 1 \
	   -auto_test -test_iter 80
