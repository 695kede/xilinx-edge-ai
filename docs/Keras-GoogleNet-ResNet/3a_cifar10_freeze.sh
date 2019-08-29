#!/bin/bash


# freeze trained graph
echo "#####################################"
echo "FREEZE GRAPH of LeNet on CIFAR10"
echo "#####################################"

rm ./freeze/cifar10/Lenet/* # remove previous results
freeze_graph --input_graph=./tf_chkpts/cifar10/LeNet/infer_graph.pb \
             --input_checkpoint=./tf_chkpts/cifar10/LeNet/float_model.ckpt \
             --input_binary=true \
             --output_graph=./freeze/cifar10/LeNet/frozen_graph.pb \
             --output_node_names=activation_4/Softmax \
             2>&1 | tee rpt/cifar10/3a_freeze_graph_LeNet.log

# check possible input/output node names
echo "#####################################"
echo "INSPECT FROZEN GRAPH of LeNet  on CIFAR10"
echo "#####################################"
decent_q inspect --input_frozen_graph=./freeze/cifar10/LeNet/frozen_graph.pb


# freeze trained graph
echo "#####################################"
echo "FREEZE GRAPH of miniVggNet  on CIFAR10"
echo "#####################################"

rm ./freeze/cifar10/miniVggNet/* #remove previous results
freeze_graph --input_graph=./tf_chkpts/cifar10/miniVggNet/infer_graph.pb \
             --input_checkpoint=./tf_chkpts/cifar10/miniVggNet/float_model.ckpt \
             --input_binary=true \
             --output_graph=./freeze/cifar10/miniVggNet/frozen_graph.pb \
             --output_node_names=activation_6/Softmax \
             2>&1 | tee rpt/cifar10/3a_freeze_graph_miniVggNet.log


# check possible input/output node names
echo "#####################################"
echo "INSPECT FROZEN GRAPH of miniVggNet on CIFAR10"
echo "#####################################"
decent_q inspect --input_frozen_graph=./freeze/cifar10/miniVggNet/frozen_graph.pb

# freeze trained graph
echo "#####################################"
echo "FREEZE GRAPH of miniGoogleNet  on CIFAR10"
echo "#####################################"

rm ./freeze/cifar10/miniGoogleNet/* #remove previous results
freeze_graph --input_graph=./tf_chkpts/cifar10/miniGoogleNet/infer_graph.pb \
             --input_checkpoint=./tf_chkpts/cifar10/miniGoogleNet/float_model.ckpt \
             --input_binary=true \
             --output_graph=./freeze/cifar10/miniGoogleNet/frozen_graph.pb \
             --output_node_names=activation_20/Softmax \
             2>&1 | tee rpt/cifar10/3a_freeze_graph_miniGoogleNet.log

# check possible input/output node names
echo "#####################################"
echo "INSPECT FROZEN GRAPH of miniGoogleNet  on CIFAR10"
echo "#####################################"
decent_q inspect --input_frozen_graph=./freeze/cifar10/miniGoogleNet/frozen_graph.pb



# freeze trained graph
echo "#####################################"
echo "FREEZE GRAPH of miniResNet  on CIFAR10"
echo "#####################################"

rm ./freeze/cifar10/miniResNet/* #remove previous results
freeze_graph --input_graph=./tf_chkpts/cifar10/miniResNet/infer_graph.pb \
             --input_checkpoint=./tf_chkpts/cifar10/miniResNet/float_model.ckpt \
             --input_binary=true \
             --output_graph=./freeze/cifar10/miniResNet/frozen_graph.pb \
             --output_node_names=activation_83/Softmax \
             2>&1 | tee rpt/cifar10/3a_freeze_graph_miniResNet.log

# check possible input/output node names
echo "#####################################"
echo "INSPECT FROZEN GRAPH of miniResNet"
echo "#####################################"
decent_q inspect --input_frozen_graph=./freeze/cifar10/miniResNet/frozen_graph.pb


echo "#####################################"
echo "FREEZE GRAPH COMPLETED"
echo "#####################################"
echo " "

