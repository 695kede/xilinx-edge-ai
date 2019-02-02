#!/bin/sh

ln -s ../test_images/ ./test_images
cp ./src/fps_main.cc ./src/main.cc
make clean
make
mv miniVggNet fps_miniVggNet

echo "./miniVggNet 1"
./fps_miniVggNet 1
echo "./miniVggNet 2"
./fps_miniVggNet 2
echo "./miniVggNet 3"
./fps_miniVggNet 3
echo "./miniVggNet 4"
./fps_miniVggNet 4
echo "./miniVggNet 5"
./fps_miniVggNet 5
echo "./miniVggNet 6"
./fps_miniVggNet 6
