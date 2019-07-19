#!/bin/sh

cp ./src/fps_tf_main.cc ./src/tf_main.cc
make clean
make
mv miniResNet fps_miniResNet

echo " "
echo "./miniResNet 1"
./fps_miniResNet 1
echo " "
echo "./miniResNet 2"
./fps_miniResNet 2
echo " "
echo "./miniResNet 3"
./fps_miniResNet 3
echo " "
echo "./miniResNet 4"
./fps_miniResNet 4
echo " "
echo "./miniResNet 5"
./fps_miniResNet 5
echo " "
echo "./miniResNet 6"
./fps_miniResNet 6
echo " "
echo "./miniResNet 7"
./fps_miniResNet 7
echo " "
echo "./miniResNet 8"
./fps_miniResNet 8
echo " "
