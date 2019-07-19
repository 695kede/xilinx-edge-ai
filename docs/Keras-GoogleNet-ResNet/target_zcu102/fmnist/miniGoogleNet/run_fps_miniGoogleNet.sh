#!/bin/sh

cp ./src/fps_tf_main.cc ./src/tf_main.cc
make clean
make
mv miniGoogleNet fps_miniGoogleNet

echo " "
echo "./miniGoogleNet 1"
./fps_miniGoogleNet 1
echo " "
echo "./miniGoogleNet 2"
./fps_miniGoogleNet 2
echo " "
echo "./miniGoogleNet 3"
./fps_miniGoogleNet 3
echo " "
echo "./miniGoogleNet 4"
./fps_miniGoogleNet 4
echo " "
echo "./miniGoogleNet 5"
./fps_miniGoogleNet 5
echo " "
echo "./miniGoogleNet 6"
./fps_miniGoogleNet 6
echo " "
echo "./miniGoogleNet 7"
./fps_miniGoogleNet 7
echo " "
echo "./miniGoogleNet 8"
./fps_miniGoogleNet 8
echo " "
