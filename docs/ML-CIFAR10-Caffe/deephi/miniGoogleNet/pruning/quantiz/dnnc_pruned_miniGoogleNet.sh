#!/bin/bash

DNNDK_ROOT=$HOME/ML/DNNDK/tools

net=miniGoogleNet
model_dir=$HOME/ML/cifar10/deephi/miniGoogleNet/pruning/quantiz/decent_output
output_dir=$HOME/ML/cifar10/deephi/miniGoogleNet/pruning/quantiz/dnnc_output

echo "Compiling network: ${net}"

dnnc --prototxt=${model_dir}/deploy.prototxt     \
       --caffemodel=${model_dir}/deploy.caffemodel \
       --output_dir=${output_dir}                  \
       --net_name=${net}                           \
       --dpu=4096FA                                 \
       --cpu_arch=arm64                            \
       --mode=debug                                \
       --save_kernel


echo " copying dpu elf file into ../../zcu102/pruned/model/arm64_4096 "
cp ${output_dir}/dpu_${net}\_*.elf  ${output_dir}/../../../zcu102/pruned/model/arm64_4096/

echo " copying the test images to be used by the ZCU102"
cp -r $HOME/ML/cifar10/input/cifar10_jpg/test ${output_dir}/../../../zcu102/test_images
mv $HOME/ML/${net}_kernel* $HOME/ML/cifar10/deephi/miniGoogleNet/pruning/quantiz/dnnc_output

cd $HOME/ML/cifar10/deephi/miniGoogleNet/zcu102/pruned
#gorce a soft link to the testing images
ln -nsf ../test_images ./test_images





