#!/bin/bash

# delete previous results
#rm -rf ./compile

# Compile
echo "##########################################################################"
echo "COMPILE WITH DNNC: LeNet on FMNIST"
echo "##########################################################################"

#LeNet
dnnc \
       --parser=tensorflow \
       --frozen_pb=./quantized_results/fmnist/LeNet/deploy_model.pb \
       --dcf=./dcf/ZCU102.dcf \
       --cpu_arch=arm64 \
       --output_dir=compile/fmnist/LeNet \
       --save_kernel \
       --mode normal \
       --net_name=LeNet \
       2>&1 | tee rpt/fmnist/5_compile_LeNet.log

#       --dpu=4096FA \

mv  ./compile/fmnist/LeNet/dpu*.elf ./target_zcu102/fmnist/LeNet/model/

# Compile
echo "##########################################################################"
echo "COMPILE WITH DNNC: miniVggNet  on FMNIST"
echo "##########################################################################"

#miniVggNet
dnnc \
       --parser=tensorflow \
       --frozen_pb=./quantized_results/fmnist/miniVggNet/deploy_model.pb \
       --dcf=./dcf/ZCU102.dcf \
       --cpu_arch=arm64 \
       --output_dir=compile/fmnist/miniVggNet \
       --save_kernel \
       --mode normal \
       --net_name=miniVggNet \
       2>&1 | tee rpt/fmnist/5_compile_miniVggNet.log

#--dpu=4096FA \
     
mv  ./compile/fmnist/miniVggNet/dpu*.elf ./target_zcu102/fmnist/miniVggNet/model/



# Compile
echo "##########################################################################"
echo "COMPILE WITH DNNC: miniGoogleNet  on FMNIST"
echo "##########################################################################"

dnnc \
       --parser=tensorflow \
       --frozen_pb=./quantized_results/fmnist/miniGoogleNet/deploy_model.pb \
       --cpu_arch=arm64 \
       --dcf=./dcf/ZCU102.dcf \
       --output_dir=compile/fmnist/miniGoogleNet \
       --save_kernel \
       --mode normal \
       --net_name=miniGoogleNet \
       2>&1 | tee rpt/fmnist/5_compile_miniGoogleNet.log

#        --dpu=4096FA \
mv  ./compile/fmnist/miniGoogleNet/dpu*.elf ./target_zcu102/fmnist/miniGoogleNet/model/


# Compile
echo "##########################################################################"
echo "COMPILE WITH DNNC: miniResNet  on FMNIST"
echo "##########################################################################"

#miniResNet
dnnc \
       --parser=tensorflow \
       --frozen_pb=./quantized_results/fmnist/miniResNet/deploy_model.pb \
       --cpu_arch=arm64 \
       --dcf=./dcf/ZCU102.dcf \
       --output_dir=compile/fmnist/miniResNet \
       --save_kernel \
       --mode normal \
       --net_name=miniResNet \
       2>&1 | tee rpt/fmnist/5_compile_miniResNet.log

#        --dpu=4096FA \
mv  ./compile/fmnist/miniResNet/dpu*.elf ./target_zcu102/fmnist/miniResNet/model/



echo "##########################################################################"
echo "COMPILATION COMPLETED  on FMNIST"
echo "##########################################################################"
echo " "
