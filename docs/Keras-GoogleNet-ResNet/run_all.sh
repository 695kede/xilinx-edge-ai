#!/bin/bash

: '
# clean up previous log files
rm -f *.log
if [ ! rpt ]; then
    mkdir rpt
else
    rm ./rpt/*.log
fi
'

##################################################################################
#organize data for Fashion-MNIST and CIFAR10
source 0_generate_images.sh

##################################################################################

# training from scratch with CIFAR10
source 1_cifar10_train.sh

# convert Keras model into TF inference graph
source 2_cifar10_Keras2TF.sh

# freeze the graphn to make predictions later
source 3a_cifar10_freeze.sh

# make predictions with frozen graph
source 3b_cifar10_evaluate_frozen_graph.sh

# quantize the CNN from 32-bit floating-point to 8-bit fixed-point
source 4a_cifar10_quant.sh

# make predictions with quantized frozen graph
source 4b_cifar10_evaluate_quantized_graph.sh

# compile ELF file for target board
source 5_cifar10_compile.sh



##################################################################################
# training from scratch with Fashion-MNIST
source 1_fmnist_train.sh


# convert Keras model into TF inference graph
source 2_fmnist_Keras2TF.sh

# freeze the graphn to make predictions later
source 3a_fmnist_freeze.sh

# make predictions with frozen graph
source 3b_fmnist_evaluate_frozen_graph.sh

# quantize the CNN from 32-bit floating-point to 8-bit fixed-point
source 4a_fmnist_quant.sh

# make predictions with quantized frozen graph
source 4b_fmnist_evaluate_quantized_graph.sh

# compile ELF file for target board
source 5_fmnist_compile.sh
