#!/bin/sh

gzip -v -d model/arm64_4096/dpu*.elf.gz

cp ./src/fps_main.cc ./src/main.cc
make clean
make all

cp alexnetBNnoLRN fps_alexnetBNnoLRN

echo " "
echo "./alexnetBNnoLRN 1"
./fps_alexnetBNnoLRN 1

echo " "
echo "./alexnetBNnoLRN 2"
./fps_alexnetBNnoLRN 2

echo " "
echo "./alexnetBNnoLRN 3"
./fps_alexnetBNnoLRN 3

echo " "
echo "./alexnetBNnoLRN 4"
./fps_alexnetBNnoLRN 4

echo " "
echo "./alexnetBNnoLRN 5"
./fps_alexnetBNnoLRN 5

echo " "
echo "./alexnetBNnoLRN 6"
./fps_alexnetBNnoLRN 6



