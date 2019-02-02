/*
 * Copyright (c) 2016-2018 DeePhi Tech, Inc.
 *
 * All Rights Reserved. No part of this source code may be reproduced
 * or transmitted in any form or by any means without the prior written
 * permission of DeePhi Tech, Inc.
 *
 * Filename: main.cc
 * Version: 1.10
 *
 * Description:
 * Sample source code showing how to deploy ResNet50 neural network on
 * DeePhi DPU@Zynq7020 platform.
 */
#include <assert.h>
#include <dirent.h>
#include <stdio.h>
#include <stdlib.h>
#include <atomic>
#include <sys/stat.h>
#include <unistd.h>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <queue>
#include <mutex> 
#include <string>
#include <vector>
#include <thread>
#include <mutex>
#include <dnndk/dnndk.h>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;
using namespace std::chrono;

int threadnum;
//#define RESNET50_WORKLOAD_CONV (7.71f)
//#define RESNET50_WORKLOAD_FC (4.0f / 1000) 
 

/* 3.16GOP times calculation for GoogLeNet CONV */
//#define GOOGLENET_WORKLOAD_CONV (3.16f)
/* (2.048/1000)GOP times calculation for GoogLeNet FC */
//#define GOOGLENET_WORKLOAD_FC (2.048f / 1000)

#define KERNEL_CONV      "miniGoogleNet_0"
#define KERNEL_FC        "miniGoogleNet_2"
#define CONV_INPUT_NODE  "conv1_3x3_s1"
#define CONV_OUTPUT_NODE "inception_11a_output" 
#define FC_INPUT_NODE    "loss_classifier"
#define FC_OUTPUT_NODE   "loss_classifier"

const string baseImagePath = "./images/";


#ifdef SHOWTIME
#define _T(func)                                                              \
    {                                                                         \
        auto _start = system_clock::now();                                    \
        func;                                                                 \
        auto _end = system_clock::now();                                      \
        auto duration = (duration_cast<microseconds>(_end - _start)).count(); \
        string tmp = #func;                                                   \
        tmp = tmp.substr(0, tmp.find('('));                                   \
        cout << "[TimeTest]" << left << setw(30) << tmp;                      \
        cout << left << setw(10) << duration << "us" << endl;                 \
    }
#else
#define _T(func) func;
#endif
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
            if ((ext == "JPEG") || (ext == "jpeg") || (ext == "JPG") || (ext == "jpg") ||
                (ext == "PNG") || (ext == "png")) {
                images.push_back(name);
            }
        }
    }

    closedir(dir);
}

/**
 * @brief load kinds from file to a vector
 *
 * @param path - path of the kind file
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
 * @brief softmax operation
 *
 * @param data - pointer to input buffer
 * @param size - size of input buffer
 * @param result - calculation result
 *
 * @return none
 */
void CPUCalcSoftmax(const float *data, size_t size, float *result) {
    assert(data && result);
    double sum = 0.0f;

    for (size_t i = 0; i < size; i++) {
        result[i] = exp(data[i]);
        sum += result[i];
    }

    for (size_t i = 0; i < size; i++) {
        result[i] /= sum;
    }
}

/**H
 * @brief Get top k results according to its probability
 *
 * @param d - pointer to input data
 * @param size - size of input data
 * @param k - calculation result
 * @param vkinds - vector of kinds
 *
 * @return none
 */
void TopK(const float *d, int size, int k, vector<string> &vkind) {
    assert(d && size > 0 && k > 0);
    priority_queue<pair<float, int>> q;

    for (auto i = 0; i < size; ++i) {
        q.push(pair<float, int>(d[i], i));
    }

    for (auto i = 0; i < k; ++i) {
        pair<float, int> ki = q.top();
        //printf("[Top]%d prob = %-8f  name = %s\n", i, d[ki.second], vkind[ki.second].c_str());
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
vector<string> kinds, images;
void runGoogLeNet(DPUTask *taskConv, DPUTask *taskFC, Mat img) {
    assert(taskConv && taskFC);
    _T(dpuRunTask(taskConv));

}

/*
 * @brief  - Entry of face detection using Densebox
 *
 * @param kernel - point to DPU Kernel
 */
void classifyEntry(DPUKernel *kernelconv, DPUKernel *kernelfc) {

    ListImages(baseImagePath, images);
    if (images.size() == 0) {
        cerr << "\nError: Not images exist in " << baseImagePath << endl;
        return;
    } else {
        cout << "total image : " << images.size() << endl;
    }

    /* Load all kinds words.*/
    LoadWords(baseImagePath + "labels.txt", kinds);
    if (kinds.size() == 0) {
        cerr << "\nError: Word does not exist in labels.txt." << endl;
        return;
    }

    thread workers[threadnum];

	Mat img = imread(baseImagePath + images.at(0));
    auto _start = system_clock::now(); 
    
	for (auto i = 0; i < threadnum; i++) {
		workers[i] = thread([&,i]() {
            // Create DPU Tasks from DPU Kernel
            DPUTask *taskconv = dpuCreateTask(kernelconv, 0);
            DPUTask *taskfc = dpuCreateTask(kernelfc, 0);

            for(unsigned int ind = i  ;ind < images.size();ind+=threadnum) {
				//Mat img = imread(baseImagePath + images.at(ind));
                
				//cout << "Process image index : " << ind << endl;
                // Process the image using DenseBox model
                runGoogLeNet(taskconv, taskfc, img);
                
            }

            // Destroy DPU Tasks & free resources
            dpuDestroyTask(taskconv);
            dpuDestroyTask(taskfc);
        });
    }

    // Release thread resources.
    for (auto &w : workers) {
        if (w.joinable()) w.join();
    }

       
    auto _end = system_clock::now();                                       
    auto duration = (duration_cast<microseconds>(_end - _start)).count();   
    cout << "[Time]" << duration << "us" << endl;  
    cout << "[FPS]" << images.size()*1000000.0/duration  << endl;  
}

 
int main(int argc ,char** argv) {
    if(argc == 2)
		threadnum = stoi(argv[1]);

	DPUKernel *kernelConv;
    DPUKernel *kernelFC; 

    dpuOpen();
    kernelConv = dpuLoadKernel(KERNEL_CONV);
    kernelFC = dpuLoadKernel(KERNEL_FC); 

    classifyEntry(kernelConv, kernelFC);
 
    dpuDestroyKernel(kernelConv);
    dpuDestroyKernel(kernelFC);
    dpuClose();

    return 0;
}

