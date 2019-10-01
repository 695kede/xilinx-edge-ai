// This is a demo code for using a Yolo model to do detection.
// The code is modified from examples/ssd/ssd_detect.cpp.
// Usage:
//    yolo_detect [FLAGS] model_file weights_file list_file
//
// where model_file is the .prototxt file defining the network architecture, and
// weights_file is the .caffemodel file containing the network parameters, and
// list_file contains a list of image files with the format as follows:
//    folder/img1.JPEG
//    folder/img2.JPEG
// list_file can also contain a list of video files with the format as follows:
//    folder/video1.mp4
//    folder/video2.mp4
//
#include <caffe/caffe.hpp>
#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#endif  // USE_OPENCV
#include "caffe/transformers/yolo_transformer.hpp"
#include <string>
#include <iostream>
#include <vector>

#ifdef USE_OPENCV
using namespace caffe;  // NOLINT(build/namespaces)

float sigmoid(float p){
  return 1.0/(1 + exp(-p * 1.0));
}

float overlap(float x1,float w1,float x2,float w2){
  float left = std::max(x1 - w1 / 2.0,x2 - w2/2.0);
  float right = std::min(x1 + w1 / 2.0,x2 + w2/2.0);
  return right - left;
}

float cal_iou(vector<float> box, vector<float> truth){
  float w = overlap(box[0],box[2],truth[0],truth[2]);
  float h = overlap(box[1],box[3],truth[1],truth[3]);
  if (w<0 || h<0)
    return 0;
  float inter_area = w * h;
  float union_area = box[2] * box[3] + truth[2] * truth[3] - inter_area;
  return inter_area * 1.0 / union_area;
}

vector<vector<float>> apply_nms(vector<vector<float>>boxes , float thres){
  vector<pair<int, float>> order(boxes.size());
  for(size_t i = 0; i < boxes.size(); ++i){
    order[i].first = i;
    order[i].second = boxes[i][7];
  }
  sort(order.begin(), order.end(),[](const pair<int, float>& ls, const pair<int, float>& rs) {return ls.second > rs.second;});
  vector<vector<float>> result;
  vector<bool> exist_box(boxes.size(), true);
  for(size_t _i = 0; _i < boxes.size(); ++_i){
    size_t i = order[_i].first;
    if(!exist_box[i]) continue;
    result.push_back(boxes[i]);
    for(size_t _j = _i+1; _j < boxes.size(); ++_j){
      size_t j = order[_j].first;
      if(!exist_box[j]) continue;
      float ovr = cal_iou(boxes[j], boxes[i]);
      if(ovr >= thres) exist_box[j] = false;
    }
  }
  return result;
}

vector<vector<float>> correct_region_boxes(vector<vector<float>> boxes , int n ,int w ,int h ,int netw ,int neth){
  int new_w = 0;
  int new_h = 0;
  if (((float)netw/w) < ((float)neth/h)){
   new_w = netw;
   new_h = (h * netw)/w;
 }
 else {
   new_w = (w * neth)/h;
   new_h = neth;
 }
 for(int i =0; i < boxes.size() ; i++){
   boxes[i][0] = (boxes[i][0] - (netw - new_w)/2.0/netw)/((float)new_w/netw);
   boxes[i][1] = (boxes[i][1] - (neth - new_h)/2.0/neth)/((float)new_h/neth);
   boxes[i][2] *= (float)netw/new_w;
   boxes[i][3] *= (float)neth/new_h;
 }
 return boxes;
}

class Detector {
public:
  Detector(const string& model_file, const string& weights_file);

  vector<vector<float>> Detect(string file ,int w ,int h); 
private:
  shared_ptr<Net<float> > net_;
  cv::Size input_geometry_;
  int num_channels_;
};

Detector::Detector(const string& model_file, const string& weights_file){
#ifdef CPU_ONLY
  Caffe::set_mode(Caffe::CPU);
#else
  Caffe::set_mode(Caffe::GPU);
  Caffe::SetDevice(0);
#endif

  /* Load the network. */
  net_.reset(new Net<float>(model_file, TEST));
  net_->CopyTrainedLayersFrom(weights_file);

  CHECK_EQ(net_->num_inputs(), 1) << "Network should have exactly one input.";
  CHECK_EQ(net_->num_outputs(), 1) << "Network should have exactly one output.";

  Blob<float>* input_layer = net_->input_blobs()[0];
  num_channels_ = input_layer->channels();
  CHECK(num_channels_ == 3 || num_channels_ == 1)
  << "Input layer should have 1 or 3 channels.";
  input_geometry_ = cv::Size(input_layer->width(), input_layer->height());
}

vector<vector<float>> Detector::Detect(string file , int img_width , int img_height) {
  Blob<float>* input_layer = net_->input_blobs()[0];
  image img = load_image_yolo(file.c_str(), input_layer->width(), input_layer->height(), input_layer->channels());
  input_layer->set_cpu_data(img.data);
  net_->Forward();

  /* Copy the output layer to a std::vector */
  Blob<float>* result_blob = net_->output_blobs()[0];
  const float* result = result_blob->cpu_data();
  int classes =1;
  float swap[24*40][5][6];
  for(int h =0; h< 24; h++)
   for(int w =0; w<40;w++)
    for(int c =0 ; c< 30;c++)
     swap[h*40+w][c/6][c%6] = result[c*24*40+h*40+w];
   float biases[10] = {0.74,0.69,1.70,1.39,3.22,2.56,5.75,4.92,10.31,10.19};
   vector<vector<float>> boxes;
   for(int h =0; h< 24; h++){
     for(int w =0; w<40;w++){
      for(int n =0 ; n< 5;n++){
       vector<float>box,cls;
       float s=0.0;
       float x = (w+sigmoid(swap[h*40+w][n][0])) /40.0;  
       float y = (h+sigmoid(swap[h*40+w][n][1])) /24.0;  
       float ww = (exp(swap[h*40+w][n][2])*biases[2*n]) /40.0;  
       float hh = (exp(swap[h*40+w][n][3])*biases[2*n+1]) /24.0;  
       float obj_score = sigmoid(swap[h*40+w][n][4]);
       for(int p =0; p < classes; p++)
        cls.push_back(swap[h*40+w][n][5+p]);
      float large = *max_element(cls.begin(),cls.end());
      for(int p =0; p <cls.size() ;p++){
        cls[p] = exp(cls[p] -large);
        s += cls[p]; 
      }
      vector<float>::iterator biggest = max_element(cls.begin(),cls.end());
      large = * biggest;
      int max_index = distance(cls.begin(),cls.end());
      for(int p =0; p < cls.size() ; p++)
        cls[p] = cls[p] / s;
      box.push_back(x);
      box.push_back(y);
      box.push_back(ww);
      box.push_back(hh);
      box.push_back(max_index+1);
      box.push_back(obj_score);
      box.push_back(large);
      box.push_back(obj_score * large);
      if(box[5]*box[6] > 0.005)
        boxes.push_back(box);
    }
  }
} 
boxes = correct_region_boxes(boxes,boxes.size(),img_width,img_height,input_geometry_.width,input_geometry_.height);
vector<vector<float>> res = apply_nms(boxes,0.45);
return res;
}


DEFINE_string(file_type, "image",
  "The file type in the list_file. Currently support image and video.");
DEFINE_string(out_file, "",
  "If provided, store the detection results in the out_file.");
DEFINE_double(confidence_threshold, 0.01,
  "Only store detections with score higher than the threshold.");

int main(int argc, char** argv) {
  ::google::InitGoogleLogging(argv[0]);
  // Print output to stderr (while still logging)
  FLAGS_alsologtostderr = 1;

#ifndef GFLAGS_GFLAGS_H_
  namespace gflags = google;
#endif

  gflags::SetUsageMessage("Do detection using Yolo mode.\n"
    "Usage:\n"
    "   yolo_detect [FLAGS] model_file weights_file list_file\n");
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  if (argc < 4) {
    gflags::ShowUsageWithFlagsRestrict(argv[0], "examples/yolo/yolo_detect");
    return 1;
  }

  const string& model_file = argv[1];
  const string& weights_file = argv[2];
  const string& file_type = FLAGS_file_type;
  const float confidence  = FLAGS_confidence_threshold;
  // Initialize the network.
  Detector detector(model_file, weights_file);

  // Process image one by one.
  std::ifstream infile(argv[3]);
  string file;
  while (infile >> file) {
    if (file_type == "image") {
      // CHECK(img) << "Unable to decode image " << file;
      cv::Mat img = cv::imread(file, -1);
      int w = img.cols;
      int h = img.rows; 
      vector<vector<float>> results = detector.Detect(file , w , h);
      for(int i =0; i< results.size();i++){
        float xmin = (results[i][0] - results[i][2]/2.0) * w + 1;	
        float ymin = (results[i][1] - results[i][3]/2.0) * h + 1;	
        float xmax = (results[i][0] + results[i][2]/2.0) * w + 1;	
        float ymax = (results[i][1] + results[i][3]/2.0) * h + 1;	
        if(results[i][7] >confidence )
          LOG(INFO)<<results[i][7]<<" "<<xmin<<" "<<ymin<<" "<<xmax<<" "<<ymax;
      /*float xmin = max(((results[i][0] - results[i][2]/2.0) * w + 1) ,1 );	
      float ymin = max(((results[i][1] - results[i][3]/2.0) * h + 1) ,1 );	
      float xmax = min(((results[i][0] + results[i][2]/2.0) * w + 1) ,w );	
      float ymax = min(((results[i][1] + results[i][3]/2.0) * h + 1) ,h );*/	
      }  
      //cv::imwrite("detection.jpg",img);
    } else {
      LOG(FATAL) << "Unknown file_type: " << file_type;
    }
  }
  return 0;
}
#else
int main(int argc, char** argv) {
  LOG(FATAL) << "This example requires OpenCV; compile with USE_OPENCV.";
}
#endif  // USE_OPENCV
