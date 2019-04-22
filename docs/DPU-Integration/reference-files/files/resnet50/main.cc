/*
-- (c) Copyright 2018 Xilinx, Inc. All rights reserved.
--
-- This file contains confidential and proprietary information
-- of Xilinx, Inc. and is protected under U.S. and
-- international copyright and other intellectual property
-- laws.
--
-- DISCLAIMER
-- This disclaimer is not a license and does not grant any
-- rights to the materials distributed herewith. Except as
-- otherwise provided in a valid license issued to you by
-- Xilinx, and to the maximum extent permitted by applicable
-- law: (1) THESE MATERIALS ARE MADE AVAILABLE "AS IS" AND
-- WITH ALL FAULTS, AND XILINX HEREBY DISCLAIMS ALL WARRANTIES
-- AND CONDITIONS, EXPRESS, IMPLIED, OR STATUTORY, INCLUDING
-- BUT NOT LIMITED TO WARRANTIES OF MERCHANTABILITY, NON-
-- INFRINGEMENT, OR FITNESS FOR ANY PARTICULAR PURPOSE; and
-- (2) Xilinx shall not be liable (whether in contract or tort,
-- including negligence, or under any other theory of
-- liability) for any loss or damage of any kind or nature
-- related to, arising under or in connection with these
-- materials, including for any direct, or any indirect,
-- special, incidental, or consequential loss or damage
-- (including loss of data, profits, goodwill, or any type of
-- loss or damage suffered as a result of any action brought
-- by a third party) even if such damage or loss was
-- reasonably foreseeable or Xilinx had been advised of the
-- possibility of the same.
--
-- CRITICAL APPLICATIONS
-- Xilinx products are not designed or intended to be fail-
-- safe, or for use in any application requiring fail-safe
-- performance, such as life-support or safety devices or
-- systems, Class III medical devices, nuclear facilities,
-- applications related to the deployment of airbags, or any
-- other applications that could lead to death, personal
-- injury, or severe property or environmental damage
-- (individually and collectively, "Critical
-- Applications"). Customer assumes the sole risk and
-- liability of any use of Xilinx products in Critical
-- Applications, subject only to applicable laws and
-- regulations governing limitations on product liability.
--
-- THIS COPYRIGHT NOTICE AND DISCLAIMER MUST BE RETAINED AS
-- PART OF THIS FILE AT ALL TIMES.
*/

#include <assert.h>
#include <dirent.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <unistd.h>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <queue>
#include <string>
#include <vector>

/* header file OpenCV for image processing */
#include <opencv2/opencv.hpp>

/* header file for DNNDK APIs */
#include <dnndk/dnndk.h>
//#include <dnndk/dnndk.h>


using namespace std;
using namespace cv;

/* 7.71GOP computation for ResNet50 Convolution layers */
#define RESNET50_WORKLOAD_CONV (7.71f)
/* (4/1000)GOP computation for ResNet50 FC layers */
#define RESNET50_WORKLOAD_FC (4.0f / 1000)

/* DPU Kernel Name for ResNet50 CONV & FC layers */
#define KRENEL_CONV "resnet50_0"
#define KERNEL_FC "resnet50_2"

#define CONV_INPUT_NODE "conv1"
#define CONV_OUTPUT_NODE "res5c_branch2c"
#define FC_INPUT_NODE "fc1000"
#define FC_OUTPUT_NODE "fc1000"

const string baseImagePath = "../common/image500_640_480/";

/**
 * @brief put image names to a vector
 *
 * @param path - path of the image direcotry
 * @param images - the vector of image name
 *
 * @return none
 */
void ListImages(string const &path, vector<string> &images) {
    images.clear();
    struct dirent *entry;

    /*Check if path is a valid directory path. */
    struct stat s;
    lstat(path.c_str(), &s);
    if (!S_ISDIR(s.st_mode)) {
        fprintf(stderr, "Error: %s is not a valid directory!\n", path.c_str());

        exit(1);
    }

    DIR *dir = opendir(path.c_str());
    if (dir == nullptr) {
        fprintf(stderr, "Error: Open %s path failed.\n", path.c_str());
        exit(1);
    }

    while ((entry = readdir(dir)) != nullptr) {
        if (entry->d_type == DT_REG || entry->d_type == DT_UNKNOWN) {
            string name = entry->d_name;
            string ext = name.substr(name.find_last_of(".") + 1);
            if ((ext == "JPEG") || (ext == "jpeg") || (ext == "JPG") ||
                (ext == "jpg") || (ext == "PNG") || (ext == "png")) {
                images.push_back(name);
            }
        }
    }

    closedir(dir);
}

/**
 * @brief load kinds from file to a vector
 *
 * @param path - path of the kinds file
 * @param kinds - the vector of kinds string
 *
 * @return none
 */
void LoadWords(string const &path, vector<string> &kinds) {
    kinds.clear();
    fstream fkinds(path);

    if (fkinds.fail()) {
        fprintf(stderr, "Error : Open %s failed.\n", path.c_str());
        exit(1);
    }

    string kind;
    while (getline(fkinds, kind)) {
        kinds.push_back(kind);
    }

    fkinds.close();
}


/**
 * @brief Get top k results according to its probability
 *
 * @param d - pointer to input data
 * @param size - size of input data
 * @param k - calculation result
 * @param vkinds - vector of kinds
 *
 * @return none
 */
void TopK(const float *d, int size, int k, vector<string> &vkinds) {
    assert(d && size > 0 && k > 0);
    priority_queue<pair<float, int>> q;

    for (auto i = 0; i < size; ++i) {
        q.push(pair<float, int>(d[i], i));
    }

    for (auto i = 0; i < k; ++i) {
        pair<float, int> ki = q.top();
        printf("top[%d] prob = %-8f  name = %s\n", i, d[ki.second],
        vkinds[ki.second].c_str());
        q.pop();
    }
}

/**
 * @brief Compute average pooling on CPU
 *
 * @param conv - pointer to ResNet50 CONV Task
 * @param fc - pointer to ResNet50 FC Task
 *
 * @return none
 */
void CPUCalcAvgPool(DPUTask *conv, DPUTask *fc) {
    assert(conv && fc);

    /* Get output Tensor to the last Node of ResNet50 CONV Task */
    DPUTensor *outTensor = dpuGetOutputTensor(conv, CONV_OUTPUT_NODE);
    /* Get size, height, width and channel of the output Tensor */
    int tensorSize = dpuGetTensorSize(outTensor);
    int outHeight = dpuGetTensorHeight(outTensor);
    int outWidth = dpuGetTensorWidth(outTensor);
    int outChannel = dpuGetTensorChannel(outTensor);

    /* allocate memory buffer */
    float *outBuffer = new float[tensorSize];

    /* Get the input address to the first Node of FC Task */
    int8_t *fcInput = dpuGetInputTensorAddress(fc, FC_INPUT_NODE);

    /* Copy the last Node's output and convert them from IN8 to FP32 format */
    dpuGetOutputTensorInHWCFP32(conv, CONV_OUTPUT_NODE, outBuffer, tensorSize);

    /* Get scale value for the first input Node of FC task */
    float scaleFC = dpuGetInputTensorScale(fc, FC_INPUT_NODE);
    int length = outHeight * outWidth;
    float avg = (float)(length * 1.0f);

    float sum;
    for (int i = 0; i < outChannel; i++) {
        sum = 0.0f;
        for (int j = 0; j < length; j++) {
            sum += outBuffer[outChannel * j + i];
        }

        /* compute average and set into the first input Node of FC Task */
        fcInput[i] = (int8_t)(sum / avg * scaleFC);
    }

    delete[] outBuffer;
}

/**
 * @brief Run CONV Task and FC Task for ResNet50
 *
 * @param taskConv - pointer to ResNet50 CONV Task
 * @param taskFC - pointer to ResNet50 FC Task
 *
 * @return none
 */
void runResnet50(DPUTask *taskConv, DPUTask *taskFC) {
    assert(taskConv && taskFC);
    /* Mean value for ResNet50 specified in Caffe prototxt */
    vector<string> kinds, images;
    /* Load all image names.*/
    ListImages(baseImagePath, images);
    if (images.size() == 0) {
        cerr << "\nError: Not images exist in " << baseImagePath << endl;
        return;
    }

    /* Load all kinds words.*/
    LoadWords(baseImagePath + "words.txt", kinds);
        if (kinds.size() == 0) {
        cerr << "\nError: Not words exist in words.txt." << endl;
        return;
    }

    /* Get channel count of the output Tensor for FC Task  */
    int channel = dpuGetOutputTensorChannel(taskFC, FC_OUTPUT_NODE);
    float *softmax = new float[channel];
    float *FCResult = new float[channel];
    for (auto &imageName : images) {
        cout << "\nLoad image : " << imageName << endl;
        /* Load image and Set image into CONV Task with mean value */
        Mat image = imread(baseImagePath + imageName);
        dpuSetInputImage2(taskConv, CONV_INPUT_NODE, image);

        /* Launch RetNet50 CONV Task */
        cout << "\nRun ResNet50 CONV layers ..." << endl;

        dpuRunTask(taskConv);
        /* Get DPU execution time (in us) of CONV Task */
        long long timeProf = dpuGetTaskProfile(taskConv);
        cout << "  DPU CONV Execution time: " << (timeProf * 1.0f) << "us\n";
        float convProf = (RESNET50_WORKLOAD_CONV / timeProf) * 1000000.0f;
        cout << "  DPU CONV Performance: " << convProf << "GOPS\n";

        /* Compute average pooling on CPU */
        CPUCalcAvgPool(taskConv, taskFC);

        cout << "Run ResNet50 FC layers ..." << endl;

        /* Launch RetNet50 FC Task */
        dpuRunTask(taskFC);
        /* Get DPU execution time (in us) for FC Task */
        timeProf = dpuGetTaskProfile(taskFC);
        cout << "  DPU FC Execution time: " << (timeProf * 1.0f) << "us\n";
        float fcProf = (RESNET50_WORKLOAD_FC / timeProf) * 1000000.0f;
        cout << "  DPU FC Performance: " << fcProf << "GOPS\n";
        DPUTensor *outTensor = dpuGetOutputTensor(taskFC, FC_OUTPUT_NODE);
        int8_t *outAddr = dpuGetTensorAddress(outTensor);
        float convScale=dpuGetOutputTensorScale(taskFC, FC_OUTPUT_NODE,  0);
        int size = dpuGetOutputTensorSize(taskFC, FC_OUTPUT_NODE);
        /* Calculate softmax on CPU and show TOP5 classification result */
        dpuRunSoftmax(outAddr, softmax, channel, size/channel ,convScale );
        TopK(softmax, channel, 5, kinds);

        /* Show the impage */
        cv::imshow("Classification of ResNet50", image);
        cv::waitKey(1);
    }

    delete[] softmax;
    delete[] FCResult;
}

/**
 * @brief Entry for running ResNet50 neural network
 *
 */
int main(void) {
  /* DPU Kernels/Tasks for running ResNet50 */
  DPUKernel *kernelConv;
  DPUKernel *kernelFC;
  DPUTask *taskConv;
  DPUTask *taskFC;



  /* Attach to DPU driver and prepare for running */
  dpuOpen();
  /* Create DPU Kernels for CONV & FC Nodes in ResNet50 */
  kernelConv = dpuLoadKernel(KRENEL_CONV);
  kernelFC = dpuLoadKernel(KERNEL_FC);
  /* Create DPU Tasks for CONV & FC Nodes in ResNet50 */
  taskConv = dpuCreateTask(kernelConv, 0);
  taskFC = dpuCreateTask(kernelFC, 0);

  /* Run CONV & FC Kernels for ResNet50 */
  runResnet50(taskConv, taskFC);

  /* Destroy DPU Tasks & free resources */
  dpuDestroyTask(taskConv);
  dpuDestroyTask(taskFC);
  /* Destroy DPU Kernels & free resources */
  dpuDestroyKernel(kernelConv);
  dpuDestroyKernel(kernelFC);
  /* Dettach from DPU driver & free resources */
  dpuClose();

  return 0;
}
