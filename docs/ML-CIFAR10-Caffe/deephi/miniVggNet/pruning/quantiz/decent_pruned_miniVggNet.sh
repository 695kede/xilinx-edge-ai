#!/usr/bin/sh

DNNDK_ROOT=$HOME/ML/DNNDK/tools

#working directory
work_dir=$HOME/ML/cifar10/deephi/miniVggNet/pruning/quantiz #$(pwd)
#path of float model
model_dir=${work_dir}
#output directory
output_dir=${work_dir}/decent_output

# force a soft link to the calibration data
ln -nsf ~/ML/cifar10/input/cifar10_jpg/calib ~/ML/cifar10/deephi/miniVggNet/pruning/quantiz/data/calib

# next commented 2 lines are only for documentation
## cp ${model_dir}/regular_rate_0.7/final.prototxt ${model_dir}/quantiz/q_final.prototxt
## then edit it to add the calibration images

# run DECENT
$DNNDK_ROOT/decent     quantize                                    \
	   -model ${model_dir}/rpt/q_final.prototxt \
	   -weights ${model_dir}/../transformed.caffemodel \
	   -output_dir ${output_dir} \
	   -method 1 \
	   -auto_test -test_iter 50

#	   -model ${model_dir}/regular_rate_0.7/final.prototxt \
#           -model ${model_dir}/../quantiz/float.prototxt     \
