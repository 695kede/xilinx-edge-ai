/*
 * Copyright (c) 2016-2018 DeePhi Tech, Inc.
 *
 * All Rights Reserved. No part of this source code may be reproduced
 * or transmitted in any form or by any means without the prior written
 * permission of DeePhi Tech, Inc.
 *
 * Filename: main.cc
 *
 * Description:
 * Sample source code showing how to deploy Segmentation neural network on
 * DeePhi DPU@ZCU102 platform.
 */
#include <iostream>
#include <fstream>
#include <string>
#include <opencv2/opencv.hpp>
#include <queue>
#include <cmath>
#include <SDL/SDL.h>
#include <thread>
#include <mutex>
#include <unistd.h>
#include <iomanip>
#include <chrono>
#include <atomic>

#include <dnndk/dnndk.h>
#include "time_helper.hpp"
#include "ssd_detector.hpp"
#include "prior_boxes.hpp"
#include "neon_math.hpp"
#include "time_helper.hpp"

#include <dnndk/dnndk.h>

// DPU Kernel name for SSD Convolution layers
#define KRENEL_CONV "ssd"
// DPU node name for input and output
#define CONV_INPUT_NODE "conv1_1"
#define CONV_OUTPUT_NODE_LOC "mbox_loc"
#define CONV_OUTPUT_NODE_CONF "mbox_conf"

#define BOOL_DPU_HAS_SOFTMAX 0

using namespace cv;
using namespace std;
using namespace std::chrono;
using namespace deephi;

// detection params
const float NMS_THRESHOLD = 0.5;
const float CONF_THRESHOLD = 0.3;
const int TOP_K = 400;
const int KEEP_TOP_K = 200;
int num_classes = 21;

typedef pair<int, Mat> imagePair;
typedef pair<int, MultiDetObjects> resultPair;

class paircomp
{
public:
  bool operator()(const imagePair &n1, const imagePair &n2) const
  {
    if (n1.first == n2.first)
      return n1.first > n2.first;
    return n1.first > n2.first;
  }
};

class resultcomp
{
public:
    bool operator()(const resultPair &n1, const resultPair &n2) const
    {
      if (n1.first == n2.first)
        return n1.first > n2.first;
      return n1.first > n2.first;
    }
};

string modeFlag;       // Flag indicating profile mode or end-to-end mode
chrono::system_clock::time_point startTime;
atomic<int> frameCnt(0);
atomic<bool> stopFlag(false);
int idxInputImage = 0; // image index of input video
int idxShowImage = 0;  // next frame index to be display

int position_x = 200;
int position_y = 200;

string videoName;      // name of input video
chrono::system_clock::time_point start_time;
// mutex for input video frame queue
mutex mtxQueueInput;

// queue for storing input video frames
queue<pair<int, Mat>> queueInput;


// mutex for display queue
mutex mtxQueueShow;
mutex mtxResultOut;
// queue for displaying images after processing
priority_queue<imagePair, vector<imagePair>, paircomp> queueShow;
priority_queue<resultPair, vector<resultPair>, resultcomp> resultOut;

/**
 * @brief Calculate softmax on CPU
 *
 * @param src - pointer to int8_t DPU data to be calculated
 * @param size - size of input int8_t DPU data
 * @param scale - scale to miltiply to transform DPU data from int8_t to float
 * @param dst - pointer to float result after softmax
 *
 * @return none
 */
void doImg(MultiDetObjects results, Mat& img){
  for (size_t i = 0; i < results.size(); ++i) {
    int label = get<0>(results[i]);
    int xmin = get<2>(results[i]).x * img.cols;
    int ymin = get<2>(results[i]).y * img.rows;
    int xmax = xmin + (get<2>(results[i]).width) * img.cols;
    int ymax = ymin + (get<2>(results[i]).height) * img.rows;
    float confidence = get<1>(results[i]);
    xmin = std::min(std::max(xmin, 0), img.cols);
    xmax = std::min(std::max(xmax, 0), img.cols);
    ymin = std::min(std::max(ymin, 0), img.rows);
    ymax = std::min(std::max(ymax, 0), img.rows);
//TODO: update lables to include 20 classes
    if (label == 1) {
      rectangle(img, Point(xmin, ymin), Point(xmax, ymax), Scalar(0, 128, 0), 1,
                1, 0);
    } else if (label == 2) {
      rectangle(img, Point(xmin, ymin), Point(xmax, ymax), Scalar(128, 0, 0), 1,
                1, 0);
    } else if (label == 3) {
      rectangle(img, Point(xmin, ymin), Point(xmax, ymax), Scalar(0, 0, 128), 1,
                1, 0);
    }else if (label == 4) {
      rectangle(img, Point(xmin, ymin), Point(xmax, ymax), Scalar(64, 64, 64), 1,
                1, 0);
    }else if (label == 5) {
      rectangle(img, Point(xmin, ymin), Point(xmax, ymax), Scalar(128, 64, 0), 1,
                1, 0);
    }else if (label == 6) {
      rectangle(img, Point(xmin, ymin), Point(xmax, ymax), Scalar(0, 64, 128), 1,
                1, 0);
    }else if (label == 7) {
      rectangle(img, Point(xmin, ymin), Point(xmax, ymax), Scalar(256, 0, 256), 1,
                1, 0);
    }else if (label == 8) {
      rectangle(img, Point(xmin, ymin), Point(xmax, ymax), Scalar(256, 128, 128), 1,
                1, 0);
    }else if (label == 9) {
      rectangle(img, Point(xmin, ymin), Point(xmax, ymax), Scalar(128, 128, 128), 1,
                1, 0);
    }else if (label == 10) {
      rectangle(img, Point(xmin, ymin), Point(xmax, ymax), Scalar(128, 128, 256), 1,
                1, 0);
    }else if (label == 11) {
      rectangle(img, Point(xmin, ymin), Point(xmax, ymax), Scalar(256, 64, 256), 1,
                1, 0);
    }else if (label == 12) {
      rectangle(img, Point(xmin, ymin), Point(xmax, ymax), Scalar(100, 256, 128), 1,
                1, 0);
    }else if (label == 13) {
      rectangle(img, Point(xmin, ymin), Point(xmax, ymax), Scalar(256, 256, 128), 1,
                1, 0);
    }else if (label == 14) {
      rectangle(img, Point(xmin, ymin), Point(xmax, ymax), Scalar(256, 256, 0), 1,
                1, 0);
	}else if (label == 15) {
      rectangle(img, Point(xmin, ymin), Point(xmax, ymax), Scalar(0, 256, 256), 1,
                1, 0);
	}else if (label == 16) {
      rectangle(img, Point(xmin, ymin), Point(xmax, ymax), Scalar(64, 256, 128), 1,
                1, 0);
	}else if (label == 17) {
      rectangle(img, Point(xmin, ymin), Point(xmax, ymax), Scalar(128, 256, 0), 1,
                1, 0);
	}else if (label == 18) {
      rectangle(img, Point(xmin, ymin), Point(xmax, ymax), Scalar(128, 256, 256), 1,
                1, 0);
	}else if (label == 19) {
      rectangle(img, Point(xmin, ymin), Point(xmax, ymax), Scalar(256, 256, 256), 1,
                1, 0);
	}else if (label == 20) {
      rectangle(img, Point(xmin, ymin), Point(xmax, ymax), Scalar(256, 256,64), 1,
                1, 0);
	}
	
  }
}

void CPUSoftmax(int8_t* src, int size, float scale, float* dst) {
  float sum = 0.0f;
  for (auto i = 0; i < size; ++i) {
    dst[i] = exp(src[i] * scale);
    sum += dst[i];
  }
  for (auto i = 0; i < size; ++i) {
    dst[i] /= sum;
  }
}

void CreatePriors(vector<shared_ptr<vector<float>>> *priors) {
  vector<float> variances{0.1, 0.1, 0.2, 0.2};
  vector<PriorBoxes> prior_boxes;
// vehicle detect
//TODO: need to update to 300 x 300 for new SSD

prior_boxes.emplace_back(PriorBoxes{
	  300, 300, 38, 38, variances, {30}, {60}, {2}, 0.5, 8.0, 8.0});
prior_boxes.emplace_back(PriorBoxes{
	  300, 300, 19, 19, variances, {60.0}, {111.0}, {2, 3}, 0.5, 16, 16});
prior_boxes.emplace_back(PriorBoxes{
	  300, 300, 10, 10, variances, {111.0}, {162.0}, {2, 3}, 0.5, 32, 32});
prior_boxes.emplace_back(PriorBoxes{
	  300, 300, 5, 5, variances, {162.0}, {213.0}, {2, 3}, 0.5, 64, 64});
prior_boxes.emplace_back(PriorBoxes{
	  300, 300, 3, 3, variances, {213.0}, {264.0}, {2}, 0.5, 100, 100});
prior_boxes.emplace_back(PriorBoxes{
	  300, 300, 1, 1, variances, {264.0}, {315.0}, {2}, 0.5, 300, 300});


/*  prior_boxes.emplace_back(PriorBoxes{
          480, 360, 60, 45, variances, {15.0, 30}, {33.0, 60}, {2}, 0.5, 8.0, 8.0});
  prior_boxes.emplace_back(PriorBoxes{
          480, 360, 30, 23, variances, {66.0}, {127.0}, {2, 3}, 0.5, 16, 16});
  prior_boxes.emplace_back(PriorBoxes{
          480, 360, 15, 12, variances, {127.0}, {188.0}, {2, 3}, 0.5, 32, 32});
  prior_boxes.emplace_back(PriorBoxes{
          480, 360, 8, 6, variances, {188.0}, {249.0}, {2, 3}, 0.5, 64, 64});
  prior_boxes.emplace_back(PriorBoxes{
          480, 360, 6, 4, variances, {249.0}, {310.0}, {2}, 0.5, 100, 100});
  prior_boxes.emplace_back(PriorBoxes{
          480, 360, 4, 2, variances, {310.0}, {372.0}, {2}, 0.5, 300, 300});
*/		  
  int num_priors = 0;
  for (auto &p : prior_boxes) {
    num_priors += p.priors().size();
  }

  priors->clear();
  priors->reserve(num_priors);
  for (auto i = 0U; i < prior_boxes.size(); ++i) {
    priors->insert(priors->end(), prior_boxes[i].priors().begin(),
                   prior_boxes[i].priors().end());
  }
}

//
// entry function to read frame image from input video file
//
void frameReader()
{
  //get timeStamp of start
  start_time = chrono::system_clock::now();
  // for profile mode, we don't read frame imagem from video
  if (modeFlag == "profile") return;
     
  VideoCapture video;
  video.open(videoName);

  while (1)
  {
    Mat img;
	// maximum iamge queue size 10
    if (queueInput.size() < 5)
    {
      if (!video.read(img))
      {
        video.set(CV_CAP_PROP_POS_FRAMES,0);
        continue;
      }

      // push a frame image into queue for afterwards processing
      mtxQueueInput.lock();
      queueInput.push(make_pair(idxInputImage++, img));
      mtxQueueInput.unlock();

    }
    else
    {
	  // sleep for 5ms if iamge queue is already full
      usleep(50000);
    }
  }
}

//
// entry function for displaying frame images after process of segmentation
//
void imageDisplay()
{
  Mat img;
  
  // for end to end mode
  if (modeFlag == "end2end") {
    SDL_Surface *screen;
    SDL_Surface *image;
    SDL_Surface *opt_image;

	// disaply image with SDL
    SDL_Init(SDL_INIT_EVERYTHING);
    SDL_putenv(const_cast<char *>("SDL_VIDEO_CENTERED="));
    string str = "SDL_VIDEO_WINDOW_POS=" + to_string(position_x) + "," + to_string(position_y);
    SDL_putenv(const_cast<char *>(str.c_str()));
	
    screen = SDL_SetVideoMode(512, 256, 32, SDL_HWSURFACE);

    while (true)
    {
      mtxQueueShow.lock();
      if (queueShow.empty())
      {
        mtxQueueShow.unlock();
		// sleep when display queue is empty
        usleep(20000);
      }
      else if (idxShowImage == queueShow.top().first)
      {    
        img = queueShow.top().second;
        auto show_time = chrono::system_clock::now();
        stringstream buffer;
        auto dura = (duration_cast<microseconds>(show_time - start_time)).count();
        buffer << fixed << setprecision(2)
               << (float)queueShow.top().first / (dura / 1000000.f);
        string a = buffer.str() + "FPS";
        cv::putText(img, a, cv::Point(10, 15), 1, 1, cv::Scalar{240, 240, 240},
                    1);
        image = SDL_CreateRGBSurfaceFrom((void *)img.data,
			img.cols, img.rows, 24, img.cols * 3, 0xff0000, 0x00ff00, 0x0000ff, 0);
        opt_image = SDL_DisplayFormat(image);
        SDL_BlitSurface(opt_image, NULL, screen, NULL);
        SDL_Flip(screen);
		
        idxShowImage++;
		
		// display image and popup from queue
        queueShow.pop();
        mtxQueueShow.unlock();
		
        SDL_FreeSurface(opt_image);
        SDL_FreeSurface(image);
      }
      else
      {
        mtxQueueShow.unlock();
      }
    }
  } else if (modeFlag == "profile") { // for profile mode
    usleep(10000000);
	
	// get timeStamp of ending
    auto endTime = chrono::system_clock::now();
	
    stringstream buffer;
    auto dura = (duration_cast<microseconds>(endTime - startTime)).count();
	
	// caculate frame rate
    buffer << fixed << setprecision(2) << frameCnt / (dura / 1000000.f);
	// notify segmentation thread to stop running
    cout << "Performance: " << buffer.str() + " FPS" << endl;
    stopFlag = true;
  }
}

void RunSSD(DPUTask* task) {
  // Initializations
//  float mean[3] = {104, 117, 123};
  int8_t* loc =
          (int8_t*)dpuGetOutputTensorAddress(task, CONV_OUTPUT_NODE_LOC);
  int8_t* conf =
          (int8_t*)dpuGetOutputTensorAddress(task, CONV_OUTPUT_NODE_CONF);
  float loc_scale = dpuGetOutputTensorScale(task, CONV_OUTPUT_NODE_LOC);
  float conf_scale =
          dpuGetOutputTensorScale(task, CONV_OUTPUT_NODE_CONF);
  int size = dpuGetOutputTensorSize(task, CONV_OUTPUT_NODE_CONF);
  vector<shared_ptr<vector<float>>> priors;
  CreatePriors(&priors);

  float* conf_softmax = new float[size];

  while (1) {
   MultiDetObjects results,results_o;
   Mat img;
   pair<int, Mat> pairIndexImage; 
    
    if (modeFlag == "end2end") {
      mtxQueueInput.lock();
      if (queueInput.empty()) {
        mtxQueueInput.unlock();
        continue;
      } else {
        // get an image from input queue
        pairIndexImage = queueInput.front();
        queueInput.pop();
        mtxQueueInput.unlock();
      }
      img = pairIndexImage.second;
      dpuSetInputImage2(task, (char *) CONV_INPUT_NODE, img);
      // Run CONV Task on DPU
      auto time1 = chrono::system_clock::now();
      dpuRunTask(task);
      auto time2 = chrono::system_clock::now();
      for (int i = 0; i < size / num_classes; ++i) {
        CPUSoftmax(&conf[i * num_classes], num_classes, conf_scale, &conf_softmax[i * num_classes]);
      }
      // Post-process
     
      vector<float> th_conf(num_classes, CONF_THRESHOLD);
      SSDdetector *detector_ = new SSDdetector(num_classes, SSDdetector::CodeType::CENTER_SIZE, false,
                                               KEEP_TOP_K, th_conf, TOP_K, NMS_THRESHOLD, 1.0, priors, loc_scale);
      detector_->Detect(loc, conf_softmax, &results);
      resultPair resultPair1;
      resultPair1.first = pairIndexImage.first;
      resultPair1.second = results;
      mtxResultOut.lock();
      // store image into display queue
      resultOut.push(resultPair1);
      mtxResultOut.unlock();
      auto time5 = chrono::system_clock::now();
    }
	 Mat showMat(256, 512, CV_8UC3);
	 for (int i = 0; i < showMat.rows * showMat.cols * 3; i++) {
      showMat.data[i] = img.data[i];
    }
	while (true) {
      mtxResultOut.lock();
      if (resultOut.empty())
      {
        mtxResultOut.unlock();
        // sleep when display queue is empty
        usleep(20000);
      }
      else if (resultOut.top().first == pairIndexImage.first)
      {
      results_o = resultOut.top().second;
      resultOut.pop();
      doImg(results_o, showMat);
      pairIndexImage.second = showMat;
      mtxQueueShow.lock();
        // store image into display queue
      queueShow.push(pairIndexImage);
      mtxQueueShow.unlock();
      mtxResultOut.unlock();
	  break;
      } else{
        mtxResultOut.unlock();
      }
  }
 }
      delete[] conf_softmax;
}


int main(int argc, char **argv)
{
  if ( argc != 5) {
    cout << "ERR: please specify input video, running mode and window position" << endl;
    return 1;
  }

  // get name for input video file  
  videoName = argv[1];
  
  // get mode flag: 
  // "profile" - for evaluate performance in FPS
  // "end2end" - for whole process including reading video frame image and displaying result
  modeFlag = argv[2];

  //set window osition
  position_x = atoi(argv[3]);
  position_y = atoi(argv[4]);
  if (modeFlag != "end2end" && modeFlag != "profile" ) {
    return 0;
  }
  // open DPU device
  dpuOpen();
  DPUKernel *ssdkernel;

  ssdkernel = dpuLoadKernel(KRENEL_CONV);

  // load DPU Kernel for segmentation network

  // create multi-threading mode DPU application
  int thread_nums = 1;
  vector<DPUTask *> ssdtask(1);

  generate(ssdtask.begin(), ssdtask.end(), std::bind(dpuCreateTask, ssdkernel, 0));

  array<thread, 3> threads =
          {
                  // create 3 threads for running segmentation network on DPU
                  thread(RunSSD, ssdtask[0]),

                  // create thread for reading video frame
                  thread(frameReader),

                  // create one thread for displaying image
                  thread(imageDisplay)
          };

  for (int i = 0; i < 6; i++)
  {
    threads[i].join();
  }

  // release DPU Kernel & Task resources
  for_each(ssdtask.begin(), ssdtask.end(), dpuDestroyTask);
  dpuDestroyKernel(ssdkernel);

  dpuClose();

  return 0;
}
