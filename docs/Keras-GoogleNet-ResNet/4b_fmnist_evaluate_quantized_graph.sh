#!/bin/bash

#conda activate decent_q3

echo "##########################################################################"
echo "EVALUATE QUANTIZED GRAPH of LeNet on FMNIST"
echo "##########################################################################"

#LeNet
python code/eval_graph.py --graph ./quantized_results/fmnist/LeNet/quantize_eval_model.pb --input_node conv2d_1_input --output_node activation_4/Softmax --gpu 0  2>&1 | tee rpt/fmnist/4b_evaluate_quantized_graph_LeNet.log


echo "##########################################################################"
echo "EVALUATE QUANTIZED GRAPH of miniVggNet  on FMNIST"
echo "##########################################################################"

#miniVggNet
python code/eval_graph.py --graph ./quantized_results/fmnist/miniVggNet/quantize_eval_model.pb --input_node conv2d_1_input --output_node activation_6/Softmax --gpu 0 2>&1 | tee rpt/fmnist/4b_evaluate_quantized_graph_miniVggNet.log


echo "##############################################################################"
echo "EVALUATE QUANTIZED GRAPH of miniGoogleNet  on FMNIST"
echo "##############################################################################"

#miniGoogleNet
python code/eval_graph.py --graph ./quantized_results/fmnist/miniGoogleNet/quantize_eval_model.pb --input_node conv2d_1_input --output_node activation_20/Softmax --gpu 0 2>&1 | tee rpt/fmnist/4b_evaluate_quantized_graph_miniGoogleNet.log



echo "##########################################################################"
echo "EVALUATE QUANTIZED GRAPH of miniResNet  on FMNIST"
echo "##########################################################################"

#miniResNet
python code/eval_graph.py --graph ./quantized_results/fmnist/miniResNet/quantize_eval_model.pb --input_node conv2d_1_input --output_node activation_83/Softmax --gpu 0 2>&1 | tee rpt/fmnist/4b_evaluate_quantized_graph_miniResNet.log


echo "##########################################################################"
echo "EVALUATE QUANTIZED GRAPH COMPLETED  on FMNIST"
echo "##########################################################################"
echo " "

