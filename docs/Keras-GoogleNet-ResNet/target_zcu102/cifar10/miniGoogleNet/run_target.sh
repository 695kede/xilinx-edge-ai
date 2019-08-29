#!/bin/bash

## unpack the test images archive
#tar -xvf ../../cifar10_test.tar.gz
#mv cifar10_test ../../
ln -nsf ../../cifar10_test ./test

## compile the executable for target board
cp ./src/top5_tf_main.cc ./src/tf_main.cc
make

## launch the executable and collect report
./miniGoogleNet 1 2>&1 | tee ./rpt/logfile_top5_miniGoogleNet.txt

## launch python script to check top-5 accuracy
python2 ./check_runtime_top5_cifar10.py -i ./rpt/logfile_top5_miniGoogleNet.txt  2>&1 | tee ./rpt/top5_accuracy_cifar10_miniGoogleNet.txt


## launch script to check fps
source ./run_fps_miniGoogleNet.sh 2>&1 | tee ./rpt/fps_cifar10_miniGoogleNet.txt

: '
# archive everything and copy back to your host PC with ssh/scp
cd ../..
tar -cvf target_zcu102.tar ./target_zcu102
gzip -v target_zcu102.tar
'
