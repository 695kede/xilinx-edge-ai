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
/*
 * Copyright (c) 2016-2018 DeePhi Tech, Inc.
 *
 * All Rights Reserved. No part of this source code may be reproduced
 * or transmitted in any form or by any means without the prior written
 * permission of DeePhi Tech, Inc.
 *
 * Filename: main.cc
 * Version: 2.08 beta
 * Description:
 * Sample source code showing how to deploy segmentation neural network on
 * DeePhi DPU platform.
 */
#include <deque>
#include <assert.h>
#include <dirent.h>
#include <stdio.h>
#include <stdlib.h>
#include <atomic>
#include <sys/stat.h>
#include <sys/time.h>
#include <cstdio>
#include <iomanip>
#include <unistd.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <mutex>
#include <opencv2/opencv.hpp>
#include <queue>
#include <string>
#include <thread>
#include <vector>

// Header files for DNNDK APIs
#include <dnndk/dnndk.h>

using namespace std;
using namespace std::chrono;
using namespace cv;

// constant for segmentation network

#define KERNEL_CONV "segmentation_0"
#define CONV_INPUT_NODE "conv1_7x7_s2"
#define CONV_OUTPUT_NODE "toplayer_p2"
string frankfurt_images = "../cityscapes/val/frankfurt/";
string lindau_images = "../cityscapes/val/lindau/";
string munster_images = "../cityscapes/val/munster/";
string append_filename = "gtFine_labelTrainIds.png";


// flags for each thread
bool is_reading = true;
bool is_running_1 = true;

queue<pair<string, Mat>> read_queue;     // read queue
mutex mtx_read_queue;     	 			 // mutex of read queue                                  
													
/**
 * @brief put image names to a vector
 *
 * @param path - path of the image direcotry
 * @param images - the vector of image name
 *
 * @return none
 */
vector<string> images;
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
 * @brief entry routine of segmentation, and put image into display queue
 *
 * @param task - pointer to Segmentation Task
 * @param is_running - status flag of the thread
 *
 * @return none
 */
void runSegmentation(DPUTask *task, bool &is_running) {
    // initialize the task's parameters
    DPUTensor *conv_out_tensor = dpuGetOutputTensor(task, CONV_OUTPUT_NODE);
    int outHeight = dpuGetTensorHeight(conv_out_tensor);
    int outWidth = dpuGetTensorWidth(conv_out_tensor);
    int8_t *outTensorAddr = dpuGetTensorAddress(conv_out_tensor);
    float mean[3]={73.0,82.0,72.0};
    float scale = 0.022;

    // Run detection for images in read queue
    while (is_running) {
        // Get an image from read queue
        Mat img;
        string filename;
        mtx_read_queue.lock();
        if (read_queue.empty()) {
            is_running = false;
            mtx_read_queue.unlock();
            break;
        } else {
            filename = read_queue.front().first;
            img = read_queue.front().second;
            read_queue.pop();
            mtx_read_queue.unlock();
        }


        // Set image into CONV Task with mean value
		dpuSetInputImageWithScale(task, (char *)CONV_INPUT_NODE, img, mean, scale);							   
        // Run CONV Task on DPU
        dpuRunTask(task);

        Mat segMat(outHeight, outWidth, CV_8UC1);
        for (int row = 0; row < outHeight; row++) {
            for (int col = 0; col < outWidth; col++) {
                int i = row * outWidth * 19 + col * 19;
                auto max_ind = max_element(outTensorAddr + i, outTensorAddr + i + 19);
                int posit = distance(outTensorAddr + i, max_ind);
                segMat.at<unsigned char>(row, col) = (unsigned char)(posit); //create a grayscale image with the class
            }
        }
        resize(segMat, segMat, Size(2048,1024),0,0, INTER_NEAREST);
        imwrite(filename,segMat); 
    }
}

/**
 * @brief Read frames into read queue from a video
 *
 * @param is_reading - status flag of Read thread
 *
 * @return none
 */
void Read(void) {
    cout <<"...This routine assumes all images fit into DDR Memory..." << endl;
    cout <<"Reading Frankfurt Validation Images " << endl;
    ListImages(frankfurt_images, images);
        if (images.size() == 0) {
            cerr << "\nError: No images exist in " << frankfurt_images << endl;
            return;
        }   else {
            cout << "total Frankfurt images : " << images.size() << endl;
        }
        Mat img;
        string img_result_name;
        for (unsigned int img_idx=0; img_idx<images.size(); img_idx++) {
                cout << frankfurt_images + images.at(img_idx) << endl;
                img = imread(frankfurt_images + images.at(img_idx));                                                     
                img_result_name = "results/frankfurt/" + images.at(img_idx);
                img_result_name.erase(img_result_name.end()-15,img_result_name.end());
                img_result_name.append(append_filename);
                read_queue.push(make_pair(img_result_name, img));
            }
    images.clear();
    ListImages(lindau_images, images);
    if (images.size() == 0) {
        cerr << "\nError: No images exist in " << lindau_images << endl;
        return;
    }   else {
        cout << "total Lindau images : " << images.size() << endl;
    }
    for (unsigned int img_idx=0; img_idx<images.size(); img_idx++) {
            cout << lindau_images + images.at(img_idx) << endl;
            img = imread(lindau_images + images.at(img_idx));                                                      
            img_result_name = "results/lindau/" + images.at(img_idx);
            img_result_name.erase(img_result_name.end()-15,img_result_name.end());
            img_result_name.append(append_filename);
            read_queue.push(make_pair(img_result_name, img));
        }
    images.clear();
    ListImages(munster_images, images);
    if (images.size() == 0) {
        cerr << "\nError: No images exist in " << lindau_images << endl;
        return;
    }   else {
        cout << "total Munster images : " << images.size() << endl;
    }
    for (unsigned int img_idx=0; img_idx<images.size(); img_idx++) {
            cout << munster_images + images.at(img_idx) << endl;
            img = imread(munster_images + images.at(img_idx));                                                     
            img_result_name = "results/munster/" + images.at(img_idx);
            img_result_name.erase(img_result_name.end()-15,img_result_name.end());
            img_result_name.append(append_filename);
            read_queue.push(make_pair(img_result_name, img));
        }
    images.clear();
    cout << "...processing..." << endl;
}     
       

/**
 * @brief Entry for runing Segmentation neural network
 *
 * @arg file_name[string] - path to file for detection
 *
 */
int main(int argc, char **argv) {
    DPUKernel *kernel_conv;
    DPUTask *task_conv_1;

    // Attach to DPU driver and prepare for runing
    dpuOpen();
    // Create DPU Kernels and Tasks for CONV Nodes 
    kernel_conv = dpuLoadKernel(KERNEL_CONV);
    task_conv_1 = dpuCreateTask(kernel_conv, 0);
	
    Read();
    array<thread, 1> threads = {thread(runSegmentation, task_conv_1, ref(is_running_1))};

    for (int i = 0; i < 1; ++i) {
        threads[i].join();
    }
    cout << "evaluation completed, results stored in results folder" << endl;
    // Destroy DPU Tasks and Kernels and free resources
    dpuDestroyTask(task_conv_1);
    dpuDestroyKernel(kernel_conv);
    // Detach from DPU driver and release resources
    dpuClose();

    return 0;
}
