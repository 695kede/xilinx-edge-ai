#!/bin/bash

echo "#########################################"
echo "TRAIN & EVAL LeNet on fashion MNIST"
echo "#########################################"

python code/train_fashion_mnist.py --network LeNet --weights keras_model/fmnist/LeNet --epochs 5 --init_lr 0.01 --batch_size 32 2>&1 | tee rpt/fmnist/1_train_fashion_mnist_LeNet.log


echo "###########################################"
echo "TRAIN & EVAL miniVggNet on fashion MNIST"
echo "###########################################"

python code/train_fashion_mnist.py --network miniVggNet --weights keras_model/fmnist/miniVggNet --epochs 25 --init_lr 0.01 --batch_size 64 2>&1 | tee rpt/fmnist/1_train_fashion_mnist_miniVggNet.log



echo "#############################################"
echo "TRAIN & EVAL miniGoogleNet on fashion MNIST"
echo "#############################################"

python code/train_fashion_mnist.py --network miniGoogleNet --weights keras_model/fmnist/miniGoogleNet --epochs 70 --init_lr 5e-3  --batch_size 128 2>&1 | tee rpt/fmnist/1_train_fashion_mnist_miniGoogleNet.log


echo "#############################################"
echo "TRAIN & EVAL miniResNet on fashion MNIST"
echo "#############################################"

python code/train_fashion_mnist.py --network miniResNet --weights keras_model/fmnist/miniResNet --epochs 100 --init_lr 0.1 --batch_size 128 2>&1 | tee rpt/fmnist/1_train_fashion_mnist_miniResNet.log

