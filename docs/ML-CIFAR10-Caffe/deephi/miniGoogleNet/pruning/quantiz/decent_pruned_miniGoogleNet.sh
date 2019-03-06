#!/bin/bash

ML_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && cd ../../../.. && pwd )"
export ML_DIR
$(which decent &> /dev/null) || export PATH=$HOME/ML/DNNDK/tools:$PATH
#DNNDK_ROOT=$HOME/ML/DNNDK/tools

#working directory
work_dir=$ML_DIR/deephi/miniGoogleNet/pruning/quantiz #$(pwd)
#path of float model
model_dir=${work_dir}
#output directory
output_dir=${work_dir}/decent_output

# force a soft link to the calibration data
ln -nsf $ML_DIR/input/cifar10_jpg/calib $ML_DIR/deephi/miniGoogleNet/pruning/quantiz/data/calib

# next commented 2 lines are only for documentation
## cp ${model_dir}/regular_rate_0.7/final.prototxt ${model_dir}/quantiz/q_final.prototxt
## then edit it to add the calibration images

# run DECENT
decent     quantize                                    \
	   -model ${model_dir}/rpt/q_final.prototxt \
	   -weights ${model_dir}/../transformed.caffemodel \
	   -output_dir ${output_dir} \
	   -method 1 \
	   -auto_test -test_iter 50

#	   -model ${model_dir}/regular_rate_0.7/final.prototxt \
#           -model ${model_dir}/../quantiz/float.prototxt     \
