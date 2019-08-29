#!/bin/bash


# activate DECENT_Q Python3.6 virtual environment

cd ./code

echo "##########################################################################"
echo "QUANTIZATION WITH DECENT_Q"
echo "##########################################################################"
decent_q --version


# run quantization
echo "##########################################################################"
echo "QUANTIZE LeNet on CIFAR10"
echo "##########################################################################"


# LeNet
decent_q quantize \
	 --input_frozen_graph ../freeze/cifar10/LeNet/frozen_graph.pb \
	 --input_nodes conv2d_1_input \
	 --input_shapes ?,32,32,3 \
	 --output_nodes activation_4/Softmax \
	 --output_dir ../quantized_results/cifar10/LeNet/ \
	 --method 1 \
	 --input_fn cifar10_graph_input_fn.calib_input \
	 --calib_iter 20 \
	 --gpu 0  2>&1 | tee ../rpt/cifar10/4a_quant_LeNet.log

echo "##########################################################################"
echo "QUANTIZE miniVggNet  on CIFAR10"
echo "##########################################################################"

#miniVggNet
decent_q quantize \
	 --input_frozen_graph ../freeze/cifar10/miniVggNet/frozen_graph.pb \
	 --input_nodes conv2d_1_input \
	 --input_shapes ?,32,32,3 \
	 --output_nodes activation_6/Softmax \
	 --output_dir ../quantized_results/cifar10/miniVggNet/ \
	 --method 1 \
	 --input_fn cifar10_graph_input_fn.calib_input \
	 --calib_iter 20 \
	 --gpu 0  2>&1 | tee ../rpt/cifar10/4a_quant_miniVggNet.log


echo "##########################################################################"
echo "QUANTIZE miniGoogleNet  on CIFAR10"
echo "##########################################################################"

#miniGoogleNet
decent_q quantize \
	 --input_frozen_graph ../freeze/cifar10/miniGoogleNet/frozen_graph.pb \
	 --input_nodes conv2d_1_input \
	 --input_shapes ?,32,32,3 \
	 --output_nodes activation_20/Softmax \
	 --output_dir ../quantized_results/cifar10/miniGoogleNet/ \
	 --method 1 \
	 --input_fn cifar10_graph_input_fn.calib_input \
	 --calib_iter 20 \
	 --gpu 0  2>&1 | tee ../rpt/cifar10/4a_quant_miniGoogleNet.log


echo "##########################################################################"
echo "QUANTIZE miniResNet  on CIFAR10"
echo "##########################################################################"


#miniResNet
decent_q quantize \
	 --input_frozen_graph ../freeze/cifar10/miniResNet/frozen_graph.pb \
	 --input_nodes conv2d_1_input \
	 --input_shapes ?,32,32,3 \
	 --output_nodes activation_83/Softmax \
	 --output_dir ../quantized_results/cifar10/miniResNet/ \
	 --method 1 \
	 --input_fn cifar10_graph_input_fn.calib_input \
	 --calib_iter 20 \
	 --gpu 0  2>&1 | tee ../rpt/cifar10/4a_quant_miniResNet.log



echo "##########################################################################"
echo "QUANTIZATION COMPLETED  on CIFAR10"
echo "##########################################################################"
echo " "

cd ..
