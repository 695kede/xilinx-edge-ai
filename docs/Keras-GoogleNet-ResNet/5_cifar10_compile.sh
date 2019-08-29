#!/bin/bash

# delete previous results
#rm -rf ./compile

# Compile
echo "##########################################################################"
echo "COMPILE WITH DNNC: LeNet with CIFAR10"
echo "##########################################################################"

dnnc --version

#LeNet
dnnc \
    --parser=tensorflow \
    --frozen_pb=./quantized_results/cifar10/LeNet/deploy_model.pb \
    --dcf=./dcf/ZCU102.dcf \
    --cpu_arch=arm64 \
    --output_dir=compile/cifar10/LeNet \
    --save_kernel \
    --mode normal \
    --net_name=LeNet \
    2>&1 | tee rpt/cifar10/5_compile_LeNet.log

mv  ./compile/cifar10/LeNet/dpu*.elf ./target_zcu102/cifar10/LeNet/model/


# Compile
echo "##########################################################################"
echo "COMPILE WITH DNNC: miniVggNet  with CIFAR10"
echo "##########################################################################"

#miniVggNet
dnnc \
    --parser=tensorflow \
    --frozen_pb=./quantized_results/cifar10/miniVggNet/deploy_model.pb \
    --dcf=./dcf/ZCU102.dcf \
    --cpu_arch=arm64 \
    --output_dir=compile/cifar10/miniVggNet \
    --save_kernel \
    --mode normal \
    --net_name=miniVggNet \
    2>&1 | tee rpt/cifar10/5_compile_miniVggNet.log

mv  ./compile/cifar10/miniVggNet/dpu*.elf ./target_zcu102/cifar10/miniVggNet/model/

# Compile
echo "##########################################################################"
echo "COMPILE WITH DNNC: miniGoogleNet  with CIFAR10"
echo "##########################################################################"

dnnc \
    --parser=tensorflow \
    --frozen_pb=./quantized_results/cifar10/miniGoogleNet/deploy_model.pb \
    --dcf=./dcf/ZCU102.dcf \
    --cpu_arch=arm64 \
    --output_dir=compile/cifar10/miniGoogleNet \
    --save_kernel \
    --mode normal \
    --net_name=miniGoogleNet \
    2>&1 | tee rpt/cifar10/5_compile_miniGoogleNet.log

mv  ./compile/cifar10/miniGoogleNet/dpu*.elf ./target_zcu102/cifar10/miniGoogleNet/model/


# Compile
echo "##########################################################################"
echo "COMPILE WITH DNNC: miniResNet  with CIFAR10"
echo "##########################################################################"

dnnc \
    --parser=tensorflow \
    --frozen_pb=./quantized_results/cifar10/miniResNet/deploy_model.pb \
    --dcf=./dcf/ZCU102.dcf \
    --cpu_arch=arm64 \
    --output_dir=compile/cifar10/miniResNet \
    --save_kernel \
    --mode normal \
    --net_name=miniResNet \
    2>&1 | tee rpt/cifar10/5_compile_miniResNet.log

mv  ./compile/cifar10/miniResNet/dpu*.elf ./target_zcu102/cifar10/miniResNet/model/


echo "##########################################################################"
echo "COMPILATION COMPLETED with CIFAR10"
echo "##########################################################################"
echo " "
