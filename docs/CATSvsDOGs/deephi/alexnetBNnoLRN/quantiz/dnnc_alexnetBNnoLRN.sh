#!/bin/bash

DNNDK_ROOT=$HOME/ML/DNNDK/tools

net=alexnetBNnoLRN
work_dir=$HOME/ML/cats-vs-dogs/deephi/${net}/quantiz
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


echo " copying dpu elf file into /../zcu102/baseline/model/arm64_4096 "
cp ${output_dir}/dpu_${net}\_*.elf  ${output_dir}/../../zcu102/baseline/model/arm64_4096

echo " copying the test images to be used by the ZCU102"
cp -r /home/ML/cats-vs-dogs/input/jpg/test ${output_dir}/../../zcu102/test_images

mv ${net}_kernel*.* ${work_dir}/dnnc_output

# The following lines are for documentation only. Leave them commented
# now you can create the tar archive that you will then scp/ssh copy from HOST PC to TARGET board 
cd $HOME/ML/cats-vs-dogs/deephi/${net}/zcu102/baseline
ln -nsf ../test_images ./test_images

