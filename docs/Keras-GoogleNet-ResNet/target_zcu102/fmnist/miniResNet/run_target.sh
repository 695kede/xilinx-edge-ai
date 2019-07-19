#!/bin/bash

## unpack the test images archive
#tar -xvf ../../fmnist_test.tar.gz
#mv fmnist_test ../../
ln -nsf ../../fmnist_test ./test

## compile the executable for target board
cp ./src/top5_tf_main.cc ./src/tf_main.cc
make

## launch the executable and collect report
./miniResNet 1 2>&1 | tee ./rpt/logfile_top5_miniResNet.txt

## launch python script to check top-5 accuracy
python2 ./check_runtime_top5_fashionmnist.py -i ./rpt/logfile_top5_miniResNet.txt  2>&1 | tee ./rpt/top5_accuracy_fmnist_miniResNet.txt


## launch script to check fps

source run_fps_miniResNet.sh 2>&1 | tee ./rpt/fps_fmnist_miniResNet.txt

: '
# archive everything and copy back to your host PC with ssh/scp
cd ../..
tar -cvf target_zcu102.tar ./target_zcu102
gzip -v target_zcu102.tar
'
