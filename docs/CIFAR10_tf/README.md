<div style="page-break-after: always;"></div>

<table style="width:100%">
  <tr>
    <th width="100%" colspan="6"><img src="https://www.xilinx.com/content/dam/xilinx/imgs/press/media-kits/corporate/xilinx-logo.png" width="30%"/><h1>CIFAR10 Classification with TensorFlow</h2>
</th>
  </tr>

</table>

# Introduction

This tutorial introduces the deep neural network development kit (DNNDK) v3.0 TensorFlow design process and describes the process of creating a compiled `.elf` file that is ready for deployment on the Xilinx® deep learning processor unit (DPU) accelerator from a simple network model built using Python.

In this tutorial, you will:

+ Train and evaluate a miniVGGNet network using TensorFlow.
+ Remove the training nodes and convert the graph variables to constants. This is referred to as freezing the graph.
+ Evaluate the frozen model using the CIFAR-10 test dataset.
+ Quantize the frozen model using the DECENT_Q tool provided as part of the Xilinx DNNDK v3.0 suite.
+ Evaluate the quantized model using the CIFAR-10 test dataset.
+ Compile the quantized model using the DNNC tool provided as part of the Xilinx DNNDK v3.0 suite to create the `.elf` file ready for execution on the DPU Accelerator IP.


# The CIFAR-10 Dataset

CIFAR-10 is a publically available dataset that contains a total of 60,000 RGB images. The image size is 32-pixels x 32-pixels x 8-bits per color channel. This dataset is a good starting point for understanding machine learning, but the small image size of 32 x 32 means that they are not very useful for real-world applications. The complete dataset of 60,000 images is usually divided into 50,000 images for training and 10,000 images for validation.

There are a total of 10 mutually exclusive classes (or labels) as shown in the following figure:

![Alt text](./img/cifar10.png?raw=true "CIFAR-10 labels and example images")


# The miniVGGNet Convolution Neural Network

miniVGGNet was developed by Dr. Adrian Rosebrock as a teaching aid. As the name suggests, it is a reduced version of the original VGG network. It has a simple structure that consists of a single block of layers that has been repeated twice. The layers are as follows:

+ Convolution layer
+ Batch normalization layer
+ Convolution layer
+ Batch normalization layer
+ Maxpooling layer
+ Dropout layer

After the second block of `conv-bn-maxpool` layers, there are the usual dense (or, fully-connected) layers that is typical of many classification networks. These layers are as follows:

+ Flatten layer
+ Dense (or, fully-connected) layer
+ Batch normalization layer
+ Dropout layer
+ Dense (or, fully-connected) layer


![Alt text](./img/minivggnet.png?raw=true "miniVGGNet")


# Prerequistes

+ Ubuntu 16.04 platform with the following tools installed:
  + Anaconda 3
  + CUDA 9.0 and cuDNN 7.0.5
  + DNNDK v3.0
  + Jupyter Notebooks. To run the tutorial using Jupyter Notebooks, install it using Anaconda.
+ DECENT_Q. The version must be compatible with Python 3 since most scripts in this tutorial are based on Python3. For more information, refer to the <a href="https://www.xilinx.com/support/documentation/user_guides/ug1327-dnndk-user-guide.pdf">DNNDK v3.0 User guide (UG1327)</a>.
+ Experience using Python3.
+ Familiarity with machine learning principles.


# Getting Started

Clone or download the GitHub repository to your local machine where you have installed the required tools.


## Running the Tutorial using Jupyter Notebooks

The complete flow is illustrated using two Jupyter notebooks, which can either be viewed directly in GitHub or downloaded and executed as tutorials.

1. If you are not familiar with TensorFlow and would like to have an example design, see <a href="cifar10_tf_1.ipynb">Part 1 -  Training and evaluation of the network</a>.

2. If you are familiar with TensorFlow, run all the cells in Part 1 and then proceed with <a href="cifar10_tf_2.ipynb">Part 2 -  Preparing for Deployment</a>, which describes the flows of the DNNDK v3.0 tool suite.

 >**:information_source: TIP** View and/or run the Jupyter Notebooks to familiarize yourself with the complete flow before running the tutorial using scripts.


## Running the Tutorial using Scripts

Jupyter Notebooks provide a step-by-step guide to working with TensorFlow and DNNDK. However, a real-world design is executed using a mix of Python and Linux shell scripts. This tutorial provides all of the necessary scripts to execute a design.


1. Open a terminal.
3. Run the `cd` command to move into the repository you created.
2. Run the complete flow using the ``source ./run_all.sh`` script.
5. Run each step individually using the following scripts as required:

    + ``source ./dnload.sh``  - Downloads the CIFAR-10 source images in `.png` format and creates lists of images for calibration and evaluation.
    + ``source ./train.sh``   - Executes the training and evaluates the network using the CIFAR-10 handwritten digits dataset.
    + ``source ./freeze.sh``  - Creates the frozen graph in `.pb` format.
    + ``source ./evaluate_frozen_graph.sh`` - Evaluates the frozen model using the CIFAR-10 `.png` images.
    + ``source ./quant.sh``   - Runs the DECENT_q for quantization.
    + ``source ./evaluate_quantized_graph.sh`` - Evaluates the quantized model using the CIFAR-10 `.png` images.
    + ``source ./compile.sh`` - Runs the DNNC to create the `.elf` file.

>**:pushpin: NOTES**
>1. The ``compile.sh`` script targets the B4096 DPU. You might have to change this depending on which Zynq&reg; family device you are targeting.
>2. Most scripts, like the ``conda activate decent_q3`` script, contain references to the Python virtual environments that are handled by Anaconda. The names of the virtual environment must be modified to match your system.

<hr/>
<p align="center"><sup>Copyright&copy; 2019 Xilinx</sup></p>
