/*
 * Copyright (c) 2016-2018 DeePhi Tech, Inc.
 *
 * All Rights Reserved. No part of this publication may be reproduced
 * or transmitted in any form or by any means without the prior written
 * permission of DeePhi Tech, Inc.
 *
 * Filename: tf_main.cc
 * Version: 1.10
 *
 * Description :
 * Sample source code showing how to deploy KERAS LeNet neural network on
 * DeePhi DPU on ZCU102 board.
 *
 * modified by Daniele Bagni on 11 July 2019
 */

#include <assert.h>
#include <dirent.h>
#include <stdio.h>
#include <stdlib.h>
#include <atomic> //DB
#include <sys/stat.h>
#include <unistd.h> //DB
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <iostream>
#include <iomanip> //DB
#include <queue>
#include <mutex>  //DB
#include <string>
#include <vector>
#include <thread> //DB
#include <opencv2/opencv.hpp>
/* header files for DNNDK APIs */
#include <dnndk/dnndk.h>

using namespace std;
using namespace cv;
using namespace std::chrono;

int threadnum;

#define KERNEL_CONV "LeNet_0"

#define CONV_INPUT_NODE "conv2d_2_convolution"
#define CONV_OUTPUT_NODE "dense_2_MatMul"

const string baseImagePath = "./test/";


//#define SHOWTIME

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


/*List all images's name in path.*/
void ListImages(std::string const &path, std::vector<std::string> &images) {
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
      std::string name = entry->d_name;
      std::string ext = name.substr(name.find_last_of(".") + 1);
      if ((ext == "JPEG") || (ext == "jpeg") || (ext == "JPG") || (ext == "jpg") ||
          (ext == "bmp") ||  (ext == "BMP") || (ext == "PNG") || (ext == "png")) {
        images.push_back(name);
      }
    }
  }

  closedir(dir);
}

/*Load all kinds*/
void LoadWords(std::string const &path, std::vector<std::string> &kinds) {
  kinds.clear();
  std::fstream fkinds(path);
  if (fkinds.fail()) {
    fprintf(stderr, "Error : Open %s failed.\n", path.c_str());
    exit(1);
  }
  std::string kind;
  while (getline(fkinds, kind)) {
    kinds.push_back(kind);
  }

  fkinds.close();
}

/* Calculate softmax.*/
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

/* Get top k results and show them based on kinds. */
void TopK(const float *d, int size, int k, std::vector<std::string> &vkind) {
  assert(d && size > 0 && k > 0);
  std::priority_queue<std::pair<float, int>> q;

  for (auto i = 0; i < size; ++i) {
    q.push(std::pair<float, int>(d[i], i));
  }

  for (auto i = 0; i < k; ++i)
    {
      std::pair<float, int> ki = q.top();
      int real_ki = ki.second;
      fprintf(stdout, "top[%d] prob = %-8f  name = %s\n", i, d[ki.second], vkind[real_ki].c_str());
      q.pop();
  }
}

void central_crop(const Mat& image, int height, int width, Mat& img) {
  int offset_h = (image.rows - height)/2;
  int offset_w = (image.cols - width)/2;
  Rect box(offset_w, offset_h, width, height);
  img = image(box);
}

void change_bgr(const Mat& image, int8_t* data, float scale, float* mean) {
  for(int i = 0; i < 3; ++i)
    for(int j = 0; j < image.rows; ++j)
      for(int k = 0; k < image.cols; ++k) {
		    data[j*image.rows*3+k*3+2-i] = (image.at<Vec3b>(j,k)[i] - (int8_t)mean[i]) * scale;
      }

}

void normalize_image(const Mat& image, int8_t* data, float scale, float* mean) {
  for(int i = 0; i < 3; ++i) {
    for(int j = 0; j < image.rows; ++j) {
      for(int k = 0; k < image.cols; ++k) {
	data[j*image.rows*3+k*3+2-i] = (float(image.at<Vec3b>(j,k)[i])/255.0 - 0.5)*2 * scale;
	//data[j*image.rows*3+k*3+2-i] = (float(image.at<Vec3b>(j,k)[i])/255.0 ) * scale;
      }
     }
   }
}


inline void set_input_image(DPUTask *task, const string& input_node, const cv::Mat& image, float* mean)
{
  //Mat cropped_img;
  DPUTensor* dpu_in = dpuGetInputTensor(task, input_node.c_str());
  float scale = dpuGetTensorScale(dpu_in);
  int width = dpuGetTensorWidth(dpu_in);
  int height = dpuGetTensorHeight(dpu_in);
  int size = dpuGetTensorSize(dpu_in);
  int8_t* data = dpuGetTensorAddress(dpu_in);

  //cout << "SET INPUT IMAGE: scale = " << scale  << endl;
  //cout << "SET INPUT IMAGE: width = " << width  << endl;
  //cout << "SET INPUT IMAGE: height= " << height << endl;
  //cout << "SET INPUT IMAGE: size  = " << size   << endl;

  normalize_image(image, data, scale, mean);
}


vector<string> kinds, images; //DB


void run_CNN(DPUTask *taskConv, Mat img)
{
  assert(taskConv);

  // Get channel count of the output Tensor
  int channel = dpuGetOutputTensorChannel(taskConv, CONV_OUTPUT_NODE);
  float *softmax = new float[channel];
  float *FCresult = new float[channel];

  float mean[3] = {0.0f, 0.0f, 0.0f};

  // Set image into Conv Task with mean value
  set_input_image(taskConv, CONV_INPUT_NODE, img, mean);

  //cout << "\nRun MNIST CONV ..." << endl;
  _T(dpuRunTask(taskConv));

  //DPUTensor *outTensor = dpuGetOutputTensor(taskConv, CONV_OUTPUT_NODE);
  //int8_t *outAddr = dpuGetTensorAddress(outTensor);
  //float convScale=dpuGetOutputTensorScale(taskConv, CONV_OUTPUT_NODE,  0);
  //int size = dpuGetOutputTensorSize(taskConv, CONV_OUTPUT_NODE);

  // Get FC result and convert from INT8 to FP32 format
  _T(dpuGetOutputTensorInHWCFP32(taskConv, CONV_OUTPUT_NODE, FCresult, channel));

  // Calculate softmax on CPU and show TOP5 classification result
  CPUCalcSoftmax(FCresult, channel, softmax);
  TopK(softmax, channel, 5, kinds);

  delete[] softmax;
  delete[] FCresult;

}


/**
 * @brief Run DPU CONV Task for Keras Net
 *
 * @param taskConv - pointer to CONV Task
 *
 * @return none
 */
void classifyEntry(DPUKernel *kernelConv)
{

  //  vector<string> kinds, images;

  /*Load all image names */
  ListImages(baseImagePath, images);
  if (images.size() == 0) {
    cerr << "\nError: Not images exist in " << baseImagePath << endl;
    return;
  } else {
    cout << "total image : " << images.size() << endl;
  }

  /*Load all kinds words.*/
  LoadWords(baseImagePath + "labels.txt", kinds);
  if (kinds.size() == 0) {
    cerr << "\nError: Not words exist in labels.txt." << endl;
    return;
  }

  /* ************************************************************************************** */
  //DB added multi-threding code

#define DPU_MODE_NORMAL 0
#define DPU_MODE_PROF   1
#define DPU_MODE_DUMP   2

  thread workers[threadnum];
  auto _start = system_clock::now();

  for (auto i = 0; i < threadnum; i++)
  {
  workers[i] = thread([&,i]()
  {

    /* Create DPU Tasks for CONV  */
    DPUTask *taskConv = dpuCreateTask(kernelConv, DPU_MODE_NORMAL); // profiling not enabled
    //DPUTask *taskConv = dpuCreateTask(kernelConv, DPU_MODE_PROF); // profiling enabled
    //enable profiling
    //int res1 = dpuEnableTaskProfile(taskConv);
    //if (res1!=0) printf("ERROR IN ENABLING TASK PROFILING FOR CONV KERNEL\n");

    for(unsigned int ind = i  ;ind < images.size();ind+=threadnum)
      {

	cout << "\nLoad image : " << images.at(ind) << endl;
	Mat img = imread(baseImagePath + images.at(ind));
	//cout << "DBG imread " << baseImagePath + images.at(ind) << endl;
	//Size sz(32,32);
	//Mat img2; resize(img, img2, sz); //DB
	run_CNN(taskConv, img);
      }
    // Destroy DPU Tasks & free resources
    dpuDestroyTask(taskConv);
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
 * @brief Entry for running LeNet neural network
 *
 * @return 0 on success, or error message dispalyed in case of failure.
 */
int main(int argc, char *argv[])
{

  DPUKernel *kernelConv;

  if(argc == 2) {
    threadnum = stoi(argv[1]);
    cout << "now running " << argv[0] << " " << argv[1] << endl;
  }
  else
      cout << "now running " << argv[0] << endl;


  /* Attach to DPU driver and prepare for running */
  dpuOpen();

  /* Create DPU Kernel for MNIST */
  kernelConv = dpuLoadKernel(KERNEL_CONV); //DB

  /* run CIFAR10 Classification  */
  classifyEntry(kernelConv);

  /* Destroy DPU Kernel  */
  dpuDestroyKernel(kernelConv);

  /* Dettach from DPU driver & release resources */
  dpuClose();

  return 0;
}
