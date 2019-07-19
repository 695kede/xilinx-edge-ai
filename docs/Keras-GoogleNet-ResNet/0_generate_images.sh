#!/bin/bash


# organize Fashion-MNIST data
python code/fashion_mnist_generate_images.py 2>&1 | tee rpt/fmnist/0_fashion_mnist_generate_images.log

# organize CIFAR10  data
python code/cifar10_generate_images.py 2>&1 | tee rpt/cifar10/0_cifar10_generate_images.log

