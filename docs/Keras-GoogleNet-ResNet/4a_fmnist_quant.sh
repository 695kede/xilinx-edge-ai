#!/bin/bash


# activate DECENT_Q Python3.6 virtual environment

cd ./code


echo "#####################################"
echo "QUANTIZE LeNet on Fashion MNIST"
echo "#####################################"


# LeNet
decent_q quantize \
	 --input_frozen_graph ../freeze/fmnist/LeNet/frozen_graph.pb \
	 --input_nodes conv2d_1_input \
	 --input_shapes ?,32,32,3 \
	 --output_nodes activation_4/Softmax \
	 --output_dir ../quantized_results/fmnist/LeNet/ \
	 --method 1 \
	 --input_fn graph_input_fn.calib_input \
	 --calib_iter 20 \
	 --gpu 0  2>&1 | tee ../rpt/fmnist/4a_quant_LeNet.log




echo "#####################################"
echo "QUANTIZE miniVggNet  on Fashion MNIST"
echo "#####################################"

#miniVggNet
decent_q quantize \
	 --input_frozen_graph ../freeze/fmnist/miniVggNet/frozen_graph.pb \
	 --input_nodes conv2d_1_input \
	 --input_shapes ?,32,32,3 \
	 --output_nodes activation_6/Softmax \
	 --output_dir ../quantized_results/fmnist/miniVggNet/ \
	 --method 1 \
	 --input_fn graph_input_fn.calib_input \
	 --calib_iter 20 \
	 --gpu 0  2>&1 | tee ../rpt/fmnist/4a_quant_miniVggNet.log




echo "#####################################"
echo "QUANTIZE miniGoogleNet  on Fashion MNIST"
echo "#####################################"

#miniGoogleNet
decent_q quantize \
	 --input_frozen_graph ../freeze/fmnist/miniGoogleNet/frozen_graph.pb \
	 --input_nodes conv2d_1_input \
	 --input_shapes ?,32,32,3 \
	 --output_nodes activation_20/Softmax \
	 --output_dir ../quantized_results/fmnist/miniGoogleNet/ \
	 --method 1 \
	 --input_fn graph_input_fn.calib_input \
	 --calib_iter 20 \
	 --gpu 0  2>&1 | tee ../rpt/fmnist/4a_quant_miniGoogleNet.log


echo "#####################################"
echo "QUANTIZE miniResNet  on Fashion MNIST"
echo "#####################################"

source ~/scripts/activate_py36_decent190708.sh #patch for decent (190624 DNNDK)

#miniResNet
decent_q quantize \
	 --input_frozen_graph ../freeze/fmnist/miniResNet/frozen_graph.pb \
	 --input_nodes conv2d_1_input \
	 --input_shapes ?,32,32,3 \
	 --output_nodes activation_83/Softmax \
	 --output_dir ../quantized_results/fmnist/miniResNet/ \
	 --method 1 \
	 --input_fn graph_input_fn.calib_input \
	 --calib_iter 20 \
	 --gpu 0  2>&1 | tee ../rpt/fmnist/4a_quant_miniResNet.log

source ~/scripts/activate_py36_decentTF.sh

echo "#####################################"
echo "QUANTIZATION COMPLETED  on Fashion MNIST"
echo "#####################################"
echo " "

cd ..
