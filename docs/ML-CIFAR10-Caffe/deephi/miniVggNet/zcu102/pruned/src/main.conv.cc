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

// modified by daniele.bagni@xilinx.com for minVggNet CNN.
// date 20 April 2018

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

#define KERNEL_CONV "miniVggNet_0"
//#define KERNEL_FC "resnet50_2"
#define CONV_INPUT_NODE "conv1"
#define CONV_OUTPUT_NODE "fc2"
//#define FC_INPUT_NODE "fc1"
//#define FC_OUTPUT_NODE "fc2"

const string baseImagePath = "./test_images/";

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
      //pair<float, int> ki = q.top();
      //printf("[Top]%d prob = %-8f  name = %s\n", i, d[ki.second], vkind[ki.second].c_str());
      q.pop();
    }
}

/**
 * @brief Run CONV Task for miniVggNet
 *
 * @param taskConv - pointer to miniVggNet CONV Task
 *
 * @return none
 */
// daniele.bagni@xilinx.com


vector<string> kinds, images;
void run_miniVggNet(DPUTask *taskConv, Mat img) {

    _T(dpuRunTask(taskConv));

}


void classifyEntry(DPUKernel *kernelconv)
{

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
        cerr << "\nError: Not words exist in words.txt." << endl;
        return;
    }

    thread workers[threadnum];

    Mat img = imread(baseImagePath + images.at(0));
    auto _start = system_clock::now(); 
    
    for (auto i = 0; i < threadnum; i++)
    {
      workers[i] = thread([&,i]()
      {
	// Create DPU Tasks from DPU Kernel
	DPUTask *taskconv = dpuCreateTask(kernelconv, 0);

	for(unsigned int ind = i  ;ind < images.size();ind+=threadnum)
	{
	  run_miniVggNet(taskconv, img);
	}
	// Destroy DPU Tasks & free resources
	dpuDestroyTask(taskconv);
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

/**
 * @brief Entry for running miniVggNet neural network
 *
 */

int main(int argc ,char** argv) {
    if(argc == 2)
		threadnum = stoi(argv[1]);

    DPUKernel *kernelConv;
    //DPUKernel *kernelFC; //DB

    dpuOpen();
    kernelConv = dpuLoadKernel(KERNEL_CONV);
    //kernelFC = dpuLoadKernel(KERNEL_FC);  //DB

    //classifyEntry(kernelConv, kernelFC); //DB
    classifyEntry(kernelConv); 
 
    dpuDestroyKernel(kernelConv);
    //dpuDestroyKernel(kernelFC); //DB
    dpuClose();

    return 0;
}

