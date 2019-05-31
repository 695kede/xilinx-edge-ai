#!/bin/bash


conda activate decent_q3


echo "#####################################"
echo "TRAIN & EVAL"
echo "#####################################"

python cifar10_train.py
