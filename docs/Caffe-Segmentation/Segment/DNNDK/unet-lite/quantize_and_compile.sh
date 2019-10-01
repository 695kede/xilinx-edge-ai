#!/usr/bin/env bash
export GPUID=0
net=segmentation

curr_dir=$(pwd) 
cd ../decent
source ./setup_decent_q.sh
chmod 777 decent* 
cd $curr_dir

#working directory
work_dir=$(pwd)
#path of float model
model_dir=decent_output
#output directory
output_dir=dnnc_output

echo "quantizing network: $(pwd)/float.prototxt"
./../decent/decent_q_segment quantize            \
          -model $(pwd)/float.prototxt     \
          -weights $(pwd)/float.caffemodel \
          -gpu $GPUID \
          -calib_iter 1000 \
          -output_dir ${model_dir} 2>&1 | tee ${model_dir}/decent_log.txt

echo "Compiling network: ${net}"

dnnc    --prototxt=${model_dir}/deploy.prototxt \
        --caffemodel=${model_dir}/deploy.caffemodel \
        --output_dir=${output_dir} \
        --net_name=${net} --dpu=4096FA \
        --cpu_arch=arm64 2>&1 | tee ${output_dir}/dnnc_log.txt