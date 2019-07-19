#!/bin/bash

echo "##############################################"
echo "KERAS to TENSORFLOW GRAPH CONVERSION for LeNet on CIFAR10"
echo "##############################################"

# convert Keras model into TF inference graph

python code/Keras2TF.py  --dataset cifar10 -n LeNet 2>&1 | tee rpt/cifar10/2_keras2TF_graph_conversion_LeNet.log

echo "###################################################"
echo "KERAS to TENSORFLOW GRAPH CONVERSION for miniVggNet  on CIFAR10"
echo "###################################################"

python code/Keras2TF.py  --dataset cifar10 -n miniVggNet 2>&1 | tee rpt/cifar10/2_keras2TF_graph_conversion_miniVggNet.log


echo "###################################################"
echo "KERAS to TENSORFLOW GRAPH CONVERSION for miniGoogleNet  on CIFAR10"
echo "###################################################"

python code/Keras2TF.py --dataset cifar10 -n miniGoogleNet 2>&1 | tee rpt/cifar10/2_keras2TF_graph_conversion_miniGoogleNet.log


echo "###################################################"
echo "KERAS to TENSORFLOW GRAPH CONVERSION for miniResNet  on CIFAR10"
echo "###################################################"

python code/Keras2TF.py --dataset cifar10 -n miniResNet 2>&1 | tee rpt/cifar10/2_keras2TF_graph_conversion_miniResNet.log

