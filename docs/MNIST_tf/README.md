<div style="page-break-after: always;"></div>

<table style="width:100%">
  <tr>
    <th width="100%" colspan="6"><img src="https://www.xilinx.com/content/dam/xilinx/imgs/press/media-kits/corporate/xilinx-logo.png" width="30%"/><h1>MNIST Classification with TensorFlow</h2>
</th>
  </tr>
  </table>

# Introduction

This tutorial introduces the deep neural network development kit (DNNDK) v3.0 TensorFlow design process and describes the process of creating a compiled `.elf` file that is ready for deployment on the Xilinx&reg; deep learning processor unit (DPU) accelerator from a simple network model built using Python.

In this tutorial, you will:

+ Train and evaluate a simple custom network using TF 1.9 (installed with DNNDK v3.0).
+ Remove the training nodes and convert the graph variables to constants. This is referred to as freezing the graph.
+ Evaluate the frozen model using the MNIST test dataset.
+ Quantize the frozen model using the DECENT_Q tool provided as part of the Xilinx DNNDK v3.0 suite.
+ Evaluate the quantized model using the MNIST test dataset.
+ Compile the quantized model to create the `.elf` file for execution on the DPU Accelerator IP.

# Prerequisites

The following are the prerequisites:

+ Ubuntu 16.04 platform with the following tools installed:
  + Anaconda 3
  + CUDA 9.0 and cuDNN 7.0.5
  + DNNDK v3.0
  + Jupyter Notebooks. To run the tutorial using Jupyter Notebooks, install it using Anaconda.
+ DECENT_Q. The version must be compatible with Python 3 since most scripts in this tutorial are based on Python3. For more information, refer to the <a href="https://www.xilinx.com/support/documentation/user_guides/ug1327-dnndk-user-guide.pdf">DNNDK v3.0 User guide (UG1327).</a>
+ Experience using Python3.
+ Familiarity with machine learning principles.

# Getting Started

Clone or download the GitHub repository to your local machine where you have installed the required tools.

## Running the Tutorial using Jupyter Notebooks

The complete flow is described in the following Jupyter Notebooks, which can either be viewed directly in GitHub or downloaded and executed as tutorials.

1. If you are not familiar with TensorFlow and would like to have an example design, see <a href="mnist_tf_1.ipynb">Part 1: Training and Evaluating the Network</a>.
2. If you are familiar with TensorFlow, run all the cells in Part 1 and then proceed with <a href="mnist_tf_2.ipynb">Part 2: Preparing for Deployment</a>, which describes the flows of the DNNDK v3.0 tool suite.

>**:information_source: TIP** View and/or run the Jupyter Notebooks to familiarize yourself with the complete flow before running the tutorial using scripts.

## Running the Tutorial using Scripts

Jupyter Notebooks provide a step-by-step guide to working with TensorFlow and DNNDK. However, a real world design is executed using a mix of Python and Linux shell scripts. This tutorial provides all of the necessary scripts to execute a design.

1. Open a terminal.
2. Run the `cd` command to move into the repository folder.
2. Run the complete flow using the ``source ./run_all.sh`` script.
3. Run each step individually using the following scripts as required:

   + ``source ./train.sh``   - Executes the training and evaluates the network using the CIFAR-10 dataset.
   + ``source ./freeze.sh``  - Creates the frozen graph in the `.pb` format.
   + ``source ./evaluate_frozen_graph.sh`` - Evaluates the quantized mode using the CIFAR-10 test dataset.
   + ``source ./quant.sh``   - Runs the `DECENT_q` for quantization.
   + ``source ./evaluate_quantized_graph.sh`` - Evaluates the quantized model using the CIFAR-10 test dataset.
   + ``source ./compile.sh`` - Runs the DNNC to create the `.elf` file.

> :pushpin: NOTES:
> 1. The `compile.sh` script targets the B1152 DPU. You might have to change this depending on the Zynq&reg; family device you are targeting.
> 2. Most of the shell scripts contain references to the Python virtual environment that was created using the `conda activate decent_q3` command during the installation of DECENT_Q. The name of your virtual environment can be different and the scripts must be modified to reflect this.

<hr/>
<p align="center"><sup>Copyright&copy; 2019 Xilinx</sup></p>
