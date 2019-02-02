#!/bin/sh


source deactivate tensorflow_p27
source deactivate caffe_p27
export LD_LIBRARY_PATH=$HOME/ML/DNNDK/cuda9_tools/:$HOME/old/tools/dnndk_libs$:/$LD_LIBRARY_PATH
export PATH=$HOME/ML/DNNDK/cuda9_tools/:$HOME/old/tools/dnndk_libs/$PATH

#ln -s $HOME/ML/DNNDK/cuda9_tools/dnnc-dpu1.3.0 $HOME/ML/DNNDK/cuda9_tools/dnnc
#ln -s $HOME/ML/DNNDK/cuda9_tools $HOME/ML/DNNDK/tools
