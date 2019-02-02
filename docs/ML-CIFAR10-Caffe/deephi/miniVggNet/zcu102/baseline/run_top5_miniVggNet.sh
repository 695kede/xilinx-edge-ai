#!/bin/sh

ln -s ../test_images/ ./test_images
cp ./src/top5_main.cc ./src/main.cc
make clean
make
mv miniVggNet top5_miniVggNet
./top5_miniVggNet 1
