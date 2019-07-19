#!/bin/sh

cp ./src/fps_tf_main.cc ./src/tf_main.cc
make clean
make
mv miniVggNet fps_miniVggNet

echo " "
echo "./miniVggNet 1"
./fps_miniVggNet 1
echo " "
echo "./miniVggNet 2"
./fps_miniVggNet 2
echo " "
echo "./miniVggNet 3"
./fps_miniVggNet 3
echo " "
echo "./miniVggNet 4"
./fps_miniVggNet 4
echo " "
echo "./miniVggNet 5"
./fps_miniVggNet 5
echo " "
echo "./miniVggNet 6"
./fps_miniVggNet 6
echo " "
echo "./miniVggNet 7"
./fps_miniVggNet 7
echo " "
echo "./miniVggNet 8"
./fps_miniVggNet 8
echo " "
