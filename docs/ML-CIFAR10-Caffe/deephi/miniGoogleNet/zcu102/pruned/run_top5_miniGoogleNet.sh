#/bin/sh


ln -s ../test_images ./images
cp ./src/top5_main.cc ./src/main.cc
make clean
make
mv miniGoogleNet top5_miniGoogleNet
./top5_miniGoogleNet 1 
