#!/bin/bash


echo "#####################################"
echo "TRAIN & EVAL LeNet on CIFAR10"
echo "#####################################"

python code/train_cifar10.py --network LeNet --weights keras_model/cifar10/LeNet --epochs 5 --init_lr 0.01 --batch_size 32 2>&1 | tee rpt/cifar10/1_train_cifar10_LeNet.log


echo "#####################################"
echo "TRAIN & EVAL miniVggNet  on CIFAR10"
echo "#####################################"

python code/train_cifar10.py --network miniVggNet --weights keras_model/cifar10/miniVggNet --epochs 25 --init_lr 0.01 --batch_size 64 2>&1 | tee rpt/cifar10/1_train_cifar10_miniVggNet.log


echo "#####################################"
echo "TRAIN & EVAL miniGoogleNet on CIFAR10"
echo "#####################################"

python code/train_cifar10.py   --network miniGoogleNet --weights keras_model/cifar10/miniGoogleNet --epochs 70 --init_lr 5e-3  --batch_size 128 2>&1 | tee rpt/cifar10/1_train_cifar10_miniGoogleNet.log


echo "#####################################"
echo "TRAIN & EVAL miniResNet  on CIFAR10"
echo "#####################################"

python code/train_cifar10.py  --network miniResNet --weights keras_model/cifar10/miniResNet --epochs 100 --init_lr 0.1 --batch_size 128 2>&1 | tee rpt/cifar10/1_train_cifar10_miniResNet.log

