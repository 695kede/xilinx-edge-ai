#!/bin/bash

tar -xvf ./test.tar

make clean

## compile the executable for target board profile method 1
cp src/tf_main_prof1.cc src/tf_main.cc
cp ./model/dpu_prof1_miniVggNet_0.elf ./model/dpu_miniVggNet_0.elf
make -f Makefile
mv ./miniVggNet ./prof1_miniVggNet
echo "./prof1_miniVggNet 1"
./prof1_miniVggNet 1 2>&1 | tee rpt/logfile_fps_prof1.txt
make clean

## compile the executable for target board profile method 2
cp src/tf_main_prof2.cc src/tf_main.cc
cp ./model/dpu_prof2_miniVggNet_0.elf ./model/dpu_miniVggNet_0.elf
make -f Makefile
mv ./miniVggNet ./prof2_miniVggNet
echo "./prof2_miniVggNet 1"
./prof2_miniVggNet 1 2>&1 | tee rpt/logfile_fps_prof2.txt
make clean
