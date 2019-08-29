#!/bin/bash

#
# Compile
echo "##########################################################################"
echo "COMPILE WITH DNNC: prof1_miniVggNet"
echo "##########################################################################"

#miniVggNet
dnnc \
    --parser=tensorflow \
    --frozen_pb=../quantized_results/cifar10/miniVggNet/deploy_model.pb \
    --dcf=../dcf/ZCU102.dcf \
    --cpu_arch=arm64 \
    --output_dir=./compile \
    --save_kernel \
    --mode normal \
    --net_name=miniVggNet

mv  ./compile/dpu_miniVggNet_0.elf ./target_zcu102/model/dpu_prof1_miniVggNet_0.elf


# Compile
echo "##########################################################################"
echo "COMPILE WITH DNNC: prof2_miniVggNet"
echo "##########################################################################"

#miniVggNet
dnnc \
    --parser=tensorflow \
    --frozen_pb=../quantized_results/cifar10/miniVggNet/deploy_model.pb \
    --dcf=../dcf/ZCU102.dcf \
    --cpu_arch=arm64 \
    --output_dir=./compile \
    --save_kernel \
    --mode debug \
    --net_name=miniVggNet

mv  ./compile/dpu_miniVggNet_0.elf ./target_zcu102/model/dpu_prof2_miniVggNet_0.elf

