#!/bin/sh

gzip -v -d model/arm64_4096/dpu*.gz

cp ./src/top2_main.cc ./src/main.cc

make clean
make all

cp alexnetBNnoLRN top2_alexnetBNnoLRN

echo " "
echo "./top2_alexnetBNnoLRN 1"
./top2_alexnetBNnoLRN 1


