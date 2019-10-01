// This is a demo code for using a Yolo model to do detection.
// The code is modified from examples/ssd/ssd_detect.cpp
// Usage:
//    yolo_run [FLAGS] model_file weights_file 
//
// where model_file is the .prototxt file defining the network architecture, and
// weights_file is the .caffemodel file containing the network parameters, 
// list file is not required. Test data is specified ImageData layer
// Add yolo_width and yolo_height in transform_param to active image preprocessing for yolo
//
// layer {
//   name: "data"
//   type: "ImageData"
//   top: "data"
//   top: "label"
//   include {
//     phase: TEST
//   }
//   transform_param {
//     mirror: false
//     yolo_height: 768
//     yolo_width: 1280
//   }
//   image_data_param {
//     source: "/home/jiangfan/test.txt"
//     batch_size: 1
//     shuffle: false
//     root_folder: "/home/jiangfan/"
//   }
// }
//
//
#include <caffe/caffe.hpp>
#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#endif  // USE_OPENCV
#include <string>
#include <vector>

#ifdef USE_OPENCV
using namespace caffe;  // NOLINT(build/namespaces)

class Detector {
public:
  Detector(const string& model_file, const string& weights_file);
  std::vector<float> Detect();
 
private:
  shared_ptr<Net<float> > net_;
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
}

std::vector<float> Detector::Detect() {
  net_->Forward();

  /* Copy the output layer to a std::vector */
  // output_blobs()[0] is label
  Blob<float>* result_blob = net_->output_blobs()[1];
  const float* result = result_blob->cpu_data();
  std::vector<float> r;
  for(int i=0; i<result_blob->count(); i++){
    r.push_back(result[i]);
  }
  return r;
}

int main(int argc, char** argv) {
  ::google::InitGoogleLogging(argv[0]);
  // Print output to stderr (while still logging)
  FLAGS_alsologtostderr = 1;

#ifndef GFLAGS_GFLAGS_H_
  namespace gflags = google;
#endif

  gflags::SetUsageMessage("Do detection using Yolo mode.\n"
        "Usage:\n"
        "   yolo_run [FLAGS] model_file weights_file\n");
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  if (argc < 3) {
    gflags::ShowUsageWithFlagsRestrict(argv[0], "examples/yolo/yolo_run");
    return 1;
  }

  const string& model_file = argv[1];
  const string& weights_file = argv[2];
  // Initialize the network.
  Detector detector(model_file, weights_file);

  std::vector<float> results = detector.Detect();
  for(size_t j=0; j<results.size(); j++){
    fprintf(stderr, "%10f,", results[j]);
  }
  return 0;
}
#else
int main(int argc, char** argv) {
  LOG(FATAL) << "This example requires OpenCV; compile with USE_OPENCV.";
}
#endif  // USE_OPENCV
