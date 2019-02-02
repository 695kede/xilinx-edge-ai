#!/bin/sh

# this script is just to remember how the entire flow MUST BE EXECUTED

cd $HOME/ML/



################################################################################################
# miniVggNet
echo "MINI VGG NET"

# Caffe Training
sh -v cifar10/caffe/caffe_flow_miniVggNet.sh                                2>&1 | tee cifar10/caffe/rpt/logfile_caffe_flow_miniVggNet.txt 

# DNNDK Quantization
echo " MINIVGGNET QUANTIZATION"
sh -v cifar10/deephi/miniVggNet/quantiz/decent_miniVggNet.sh                2>&1 | tee cifar10/deephi/miniVggNet/quantiz/rpt/logfile_decent_miniVggNet_autotest.txt 
sh -v cifar10/deephi/miniVggNet/quantiz/dnnc_miniVggNet.sh                  2>&1 | tee cifar10/deephi/miniVggNet/quantiz/rpt/logfile_dnnc_miniVggNet.txt 

# PRUNING
echo " MINIVGGNET PRUNING"
sh -v cifar10/deephi/miniVggNet/pruning/pruning_flow.sh                     2>&1 | tee cifar10/deephi/miniVggNet/pruning/rpt/logfile_pruning_flow.txt 


# Quantized PRUNED
echo " MINIVGGNET PRUNED QUANTIZATION"
sh -v cifar10/deephi/miniVggNet/pruning/quantiz/decent_pruned_miniVggNet.sh 2>&1 | tee cifar10/deephi/miniVggNet/pruning/quantiz/rpt/logfile_decent_pruned_miniVggNet.txt 
sh -v cifar10/deephi/miniVggNet/pruning/quantiz/dnnc_pruned_miniVggNet.sh   2>&1 | tee cifar10/deephi/miniVggNet/pruning/quantiz/rpt/logfile_dnnc_pruned_miniVggNet.txt 


###################################################################################################
# miniGoogleNet
echo "MINI GOOGLE NET"

# Caffe Training
sh -v cifar10/caffe/caffe_flow_miniGoogleNet.sh                    2>&1 | tee cifar10/caffe/rpt/logfile_caffe_flow_miniGoogleNet.txt 

#DNNDK Quantization
echo " MINIGOOGLENET QUANTIZATION"
sh -v cifar10/deephi/miniGoogleNet/quantiz/decent_miniGoogleNet.sh 2>&1 | tee cifar10/deephi/miniGoogleNet/quantiz/rpt/logfile_decent_miniGoogleNet_autotest.txt 
sh -v cifar10/deephi/miniGoogleNet/quantiz/dnnc_miniGoogleNet.sh   2>&1 | tee cifar10/deephi/miniGoogleNet/quantiz/rpt/logfile_dnnc_miniGoogleNet.txt 

# PRUNING
echo " MINIGOOGLENET PRUNING"
sh -v cifar10/deephi/miniGoogleNet/pruning/pruning_flow.sh                        2>&1 | tee cifar10/deephi/miniGoogleNet/pruning/rpt/logfile_pruning_flow.txt 


# Quantized PRUNED
echo " MINIGOOGLENET PRUNED QUANTIZATION"
sh -v cifar10/deephi/miniGoogleNet/pruning/quantiz/decent_pruned_miniGoogleNet.sh 2>&1 | tee cifar10/deephi/miniGoogleNet/pruning/quantiz/rpt/logfile_decent_pruded_miniGoogleNet.txt 
sh -v cifar10/deephi/miniGoogleNet/pruning/quantiz/dnnc_pruned_miniGoogleNet.sh   2>&1 | tee cifar10/deephi/miniGoogleNet/pruning/quantiz/rpt/logfile_dnnc_pruded_miniGoogleNet.txt 

