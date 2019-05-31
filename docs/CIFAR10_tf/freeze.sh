#!/bin/bash

# remove previous results
rm -rf ./freeze
mkdir ./freeze

conda activate decent_q3


# freeze trained graph
echo "#####################################"
echo "FREEZE GRAPH"
echo "#####################################"
freeze_graph --input_graph=./chkpts/inference_graph.pb \
             --input_checkpoint=./chkpts/float_model.ckpt \
             --input_binary=true \
             --output_graph=./freeze/frozen_graph.pb \
             --output_node_names=dense_1/BiasAdd


echo "#####################################"
echo "FREEZE GRAPH COMPLETED"
echo "#####################################"

# check possible input/output node names
echo "#####################################"
echo "INSPECT FROZEN GRAPH"
echo "#####################################"
decent_q inspect --input_frozen_graph=./freeze/frozen_graph.pb

