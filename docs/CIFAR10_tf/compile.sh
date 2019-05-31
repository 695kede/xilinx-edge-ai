#!/bin/bash

# delete previous results
rm -rf ./compile


conda activate decent_q3


# Compile
echo "#####################################"
echo "COMPILE WITH DNNC"
echo "#####################################"
dnnc \
       --parser=tensorflow \
       --frozen_pb=./quantize_results/deploy_model.pb \
       --dpu=4096FA \
       --cpu_arch=arm64 \
       --output_dir=compile \
       --save_kernel \
       --mode normal \
       --net_name=cifar10

echo "#####################################"
echo "COMPILATION COMPLETED"
echo "#####################################"

