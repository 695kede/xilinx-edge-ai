#!/bin/sh

cp ./src/fps_tf_main.cc ./src/tf_main.cc
make clean
make
mv LeNet fps_LeNet

echo " "
echo "./LeNet 1"
./fps_LeNet 1
echo " "
echo "./LeNet 2"
./fps_LeNet 2
echo " "
echo "./LeNet 3"
./fps_LeNet 3
echo " "
echo "./LeNet 4"
./fps_LeNet 4
echo " "
echo "./LeNet 5"
./fps_LeNet 5
echo " "
echo "./LeNet 6"
./fps_LeNet 6
echo " "
echo "./LeNet 7"
./fps_LeNet 7
echo " "
echo "./LeNet 8"
./fps_LeNet 8
echo " "
