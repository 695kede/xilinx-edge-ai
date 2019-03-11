#!/bin/bash


source deactivate tensorflow_p27
source deactivate caffe_p27
export LD_LIBRARY_PATH=$HOME/ML/DNNDK/cuda9_tools/:/usr/local/cuda-8.0/lib64/:$LD_LIBRARY_PATH
export PATH=$HOME/ML/DNNDK/cuda9_tools/:$PATH

#ln -nsf $HOME/ML/DNNDK/cuda9_tools/dnnc-dpu1.3.0 $HOME/ML/DNNDK/cuda9_toolsdnnc
#ln -nsf $HOME/ML/DNNDK/cuda9_tools $HOME/ML/DNNDK/tools
