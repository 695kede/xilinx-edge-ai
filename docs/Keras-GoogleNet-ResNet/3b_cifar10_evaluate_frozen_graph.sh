#!/bin/bash


echo "#####################################"
echo "EVALUATE FROZEN GRAPH of LeNet on CIFAR10"
echo "#####################################"

python code/eval_graph.py --dataset cifar10 --graph ./freeze/cifar10/LeNet/frozen_graph.pb --input_node conv2d_1_input --output_node activation_4/Softmax  --gpu 0  2>&1 | tee rpt/cifar10/3b_evaluate_frozen_graph_LeNet.log



echo "#####################################"
echo "EVALUATE FROZEN GRAPH of miniVggNet  on CIFAR10"
echo "#####################################"

python code/eval_graph.py --dataset cifar10 --graph ./freeze/cifar10/miniVggNet/frozen_graph.pb --input_node conv2d_1_input --output_node activation_6/Softmax  --gpu 0  2>&1 | tee rpt/cifar10/3b_evaluate_frozen_graph_miniVggNet.log


echo "#####################################"
echo "EVALUATE FROZEN GRAPH of GoogleNet  on CIFAR10"
echo "#####################################"


python code/eval_graph.py --dataset cifar10 --graph ./freeze/cifar10/miniGoogleNet/frozen_graph.pb --input_node conv2d_1_input --output_node activation_20/Softmax  --gpu 0  2>&1 | tee rpt/cifar10/3b_evaluate_frozen_graph_miniGoogleNet.log


echo "#####################################"
echo "EVALUATE FROZEN GRAPH of ResNet  on CIFAR10"
echo "#####################################"


python code/eval_graph.py --dataset cifar10 --graph ./freeze/cifar10/miniResNet/frozen_graph.pb --input_node conv2d_1_input --output_node activation_83/Softmax  --gpu 0  2>&1 | tee rpt/cifar10/3b_evaluate_frozen_graph_miniResNet.log


echo "#####################################"
echo "EVALUATE FROZEN GRAPH COMPLETED  on CIFAR10"
echo "#####################################"
echo " "
