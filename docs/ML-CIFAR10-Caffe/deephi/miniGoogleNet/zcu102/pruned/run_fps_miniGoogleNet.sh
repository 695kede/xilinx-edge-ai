#/bin/sh

ln -s ../test_images ./images

cp ./src/fps_main.cc ./src/main.cc
make clean
make
mv miniGoogleNet fps_miniGoogleNet

echo "./miniGoogle 1"
./fps_miniGoogleNet 1

echo "./miniGoogle 2"
./fps_miniGoogleNet 2

echo "./miniGoogle 3"
./fps_miniGoogleNet 3

echo "./miniGoogle 4"
./fps_miniGoogleNet 4

echo "./miniGoogle 5"
./fps_miniGoogleNet 5

echo "./miniGoogle 6"
./fps_miniGoogleNet 6






