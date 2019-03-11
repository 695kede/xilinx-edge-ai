#!/bin/sh

ML_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && cd ../../../.. && pwd )"
export ML_DIR
$(which decent &> /dev/null) || export PATH=$HOME/ML/DNNDK/tools:$PATH
#DNNDK_ROOT=$HOME/ML/DNNDK/tools

#working directory
work_dir=$ML_DIR/deephi/alexnetBNnoLRN/pruning/quantiz #$(pwd)
#path of float model
model_dir=${work_dir}
#output directory
output_dir=${work_dir}/decent_output

# soft link to the calibration data
ln -nsf $ML_DIR/input/jpg/calib $ML_DIR/deephi/alexnetBNnoLRN/pruning/quantiz/data/calib

python ${work_dir}/calibr.py -f ${work_dir}/../regular_rate_0.7/final.prototxt -i ${ML_DIR}/caffe/models/alexnetBNnoLRN/m2/header_calibr.prototxt -o ${model_dir}/q_final.prototxt


# run DECENT
decent quantize                   \
	   -model ${model_dir}/q_final.prototxt \
	   -weights ${model_dir}/../transformed.caffemodel \
	   -output_dir ${output_dir} \
	   -method 1 \
	   -auto_test -test_iter 80



