#!/bin/bash


echo "#####################################"
echo "EVALUATE FROZEN GRAPH of LeNet on Fashion MNIST"
echo "#####################################"

python code/eval_graph.py --graph ./freeze/fmnist/LeNet/frozen_graph.pb --input_node conv2d_1_input --output_node activation_4/Softmax  --gpu 0  2>&1 | tee rpt/fmnist/3b_evaluate_frozen_graph_LeNet.log


echo "#####################################"
echo "EVALUATE FROZEN GRAPH of miniVggNet  on Fashion MNIST"
echo "#####################################"

python code/eval_graph.py --graph ./freeze/fmnist/miniVggNet/frozen_graph.pb --input_node conv2d_1_input --output_node activation_6/Softmax  --gpu 0  2>&1 | tee rpt/fmnist/3b_evaluate_frozen_graph_miniVggNet.log



echo "#####################################"
echo "EVALUATE FROZEN GRAPH of GoogleNet  on Fashion MNIST"
echo "#####################################"

python code/eval_graph.py --graph ./freeze/fmnist/miniGoogleNet/frozen_graph.pb --input_node conv2d_1_input --output_node activation_20/Softmax  --gpu 0  2>&1 | tee rpt/fmnist/3b_evaluate_frozen_graph_miniGoogleNet.log


echo "#####################################"
echo "EVALUATE FROZEN GRAPH of ResNet  on Fashion MNIST"
echo "#####################################"


python code/eval_graph.py --graph ./freeze/fmnist/miniResNet/frozen_graph.pb --input_node conv2d_1_input --output_node activation_83/Softmax  --gpu 0  2>&1 | tee rpt/fmnist/3b_evaluate_frozen_graph_miniResNet.log


echo "#####################################"
echo "EVALUATE FROZEN GRAPH COMPLETED"
echo "#####################################"
echo " "
