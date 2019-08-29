#!/bin/bash


echo "##########################################################################"
echo "EVALUATE QUANTIZED GRAPH of LeNet on CIFAR10"
echo "##########################################################################"

#LeNet
python code/cifar10_eval_graph.py --dataset cifar10 --graph ./quantized_results/cifar10/LeNet/quantize_eval_model.pb --input_node conv2d_1_input --output_node activation_4/Softmax --gpu 0  2>&1 | tee rpt/cifar10/4b_evaluate_quantized_graph_LeNet.log


echo "##########################################################################"
echo "EVALUATE QUANTIZED GRAPH of miniVggNet  on CIFAR10"
echo "##########################################################################"

#miniVggNet
python code/cifar10_eval_graph.py --dataset cifar10 --graph ./quantized_results/cifar10/miniVggNet/quantize_eval_model.pb --input_node conv2d_1_input --output_node activation_6/Softmax --gpu 0 2>&1 | tee rpt/cifar10/4b_evaluate_quantized_graph_miniVggNet.log


echo "##############################################################################"
echo "EVALUATE QUANTIZED GRAPH of miniGoogleNet  on CIFAR10"
echo "##############################################################################"

#miniGoogleNet
python code/cifar10_eval_graph.py --dataset cifar10 --graph ./quantized_results/cifar10/miniGoogleNet/quantize_eval_model.pb --input_node conv2d_1_input --output_node activation_20/Softmax --gpu 0 2>&1 | tee rpt/cifar10/4b_evaluate_quantized_graph_miniGoogleNet.log

echo "##########################################################################"
echo "EVALUATE QUANTIZED GRAPH of miniResNet  on CIFAR10"
echo "##########################################################################"

#miniResNet
python code/cifar10_eval_graph.py --dataset cifar10 --graph ./quantized_results/cifar10/miniResNet/quantize_eval_model.pb --input_node conv2d_1_input --output_node activation_83/Softmax --gpu 0 2>&1 | tee rpt/cifar10/4b_evaluate_quantized_graph_miniResNet.log


echo "##########################################################################"
echo "EVALUATE QUANTIZED GRAPH COMPLETED  on CIFAR10"
echo "##########################################################################"
echo " "

