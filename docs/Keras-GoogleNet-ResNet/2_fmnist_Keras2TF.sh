#!/bin/bash



echo "##############################################"
echo "KERAS to TENSORFLOW GRAPH CONVERSION for LeNet on FMNIST"
echo "##############################################"

# convert Keras model into TF inference graph
python code/Keras2TF.py -n LeNet 2>&1 | tee rpt/fmnist/2_keras2TF_graph_conversion_LeNet.log



echo "###################################################"
echo "KERAS to TENSORFLOW GRAPH CONVERSION for miniVggNet  on FMNIST"
echo "###################################################"

python code/Keras2TF.py -n miniVggNet 2>&1 | tee rpt/fmnist/2_keras2TF_graph_conversion_miniVggNet.log



echo "###################################################"
echo "KERAS to TENSORFLOW GRAPH CONVERSION for miniGoogleNet  on FMNIST"
echo "###################################################"

python code/Keras2TF.py -n miniGoogleNet 2>&1 | tee rpt/fmnist/2_keras2TF_graph_conversion_miniGoogleNet.log


echo "###################################################"
echo "KERAS to TENSORFLOW GRAPH CONVERSION for miniResNet  on FMNIST"
echo "###################################################"

python code/Keras2TF.py -n miniResNet 2>&1 | tee rpt/fmnist/2_keras2TF_graph_conversion_miniResNet.log

