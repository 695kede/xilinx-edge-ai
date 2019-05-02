#!/bin/bash

#put here the name of your caffe distribution


##################################################################################
# set caffe dir
mkdir $HOME/caffe_tools

sudo ln -nsf /home/ubuntu/src/caffe_python_2/                           $HOME/caffe_tools/BVLC1v0-Caffe
export CAFFE_ROOT=$HOME/caffe_tools/BVLC1v0-Caffe

sudo ln -nsf /home/ubuntu/src/caffe_python_2/build/install              $CAFFE_ROOT/distribute
sudo ln -nsf /home/ubuntu/src/caffe_python_2/build/install/bin/caffe    $CAFFE_ROOT/distribute/bin/caffe.bin
sudo ln -nsf /home/ubuntu/src/caffe_python_2/build/install/bin/compute_image_mean /home/ubuntu/src/caffe_python_2/build/install/bin/compute_image_mean.bin

##################################################################################
# make all *.sh scripts to be executable
##https://askubuntu.com/questions/484718/how-to-make-a-file-executable
for file in $(find $HOME/ML/ -name *.sh); do
	chmod +x ${file}
done

##################################################################################
# set DNNDK dir, assuming $HOME/ML exists already
if [ ! -d $HOME/ML/DNNDK ]; then
	mkdir $HOME/ML/DNNDK
	cd $HOME
	cp xlnx_host_208tools_cuda9.tar.gz $HOME/ML/DNNDK

	cd $HOME/ML/DNNDK
	tar -xvf xlnx_host_208tools_cuda9.tar.gz
	cd ./cuda9_tools
	cp $HOME/ML/cats-vs-dogs/aws_scripts/libyaml* .

	ln -nsf $HOME/ML/DNNDK/cuda9_tools/dnnc-dpu1.3.0 $HOME/ML/DNNDK/cuda9_tools/dnnc
	ln -nsf $HOME/ML/DNNDK/cuda9_tools $HOME/ML/DNNDK/tools
fi

##################################################################################
# install missing packages on the AWS
source activate caffe_p27
conda install python-lmdb pydot scikit-learn
source deactivate caffe_p27


#################################################################################
#try decent and dnnc

source $HOME/ML/cats-vs-dogs/aws_scripts/aws_activate_dnndk_cuda9.sh
decent --version
dnnc --version

