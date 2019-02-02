#!/bin/bash

DNNDK_ROOT=$HOME/ML/DNNDK/tools

net=alexnetBNnoLRN
work_dir=$HOME/ML/cats-vs-dogs/deephi/alexnetBNnoLRN/pruning/quantiz
model_dir=${work_dir}/decent_output
output_dir=${work_dir}/dnnc_output

echo "Compiling network: ${net}"

$DNNDK_ROOT/dnnc --prototxt=${model_dir}/deploy.prototxt     \
       --caffemodel=${model_dir}/deploy.caffemodel \
       --output_dir=${output_dir}                  \
       --net_name=${net}                           \
       --dpu=4096FA                                \
       --cpu_arch=arm64                            \
       --mode=debug                                \
       --save_kernel

echo " copying dpu elf file into ../../zcu102/pruned/model/arm64_4096 "
cp ${output_dir}/dpu_${net}\_*.elf  ${output_dir}/../../../zcu102/pruned/model/arm64_4096

mv ~/ML/${net}_kernel* ${work_dir}/dnnc_output

echo " copying the test images to be used by the ZCU102"
cp -r $HOME/ML/cats-vs-dogs/input/jpg/test ${output_dir}/../../../zcu102/test_images
cd $HOME/ML/cats-vs-dogs/deephi/alexnetBNnoLRN/zcu102/pruned
ln -nsf ../test_images ./test_images

