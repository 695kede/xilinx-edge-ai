#include "caffe/util/convert_proto.hpp"
#include "caffe/util/insert_splits.hpp"
#include <tuple>

using namespace std;

namespace caffe {

NetParameter RemoveLayerByType(const NetParameter &model_in,
                               const string type) {
  NetParameter model_out;
  map<string, string> type_blobs;

  for (size_t i = 0; i < model_in.layer_size(); i++) {
    auto &cur_layer = model_in.layer(i);
    if (cur_layer.type() == type) {
      CHECK(!(cur_layer.bottom_size() > 1 && cur_layer.top_size() > 1))
          << "Remove Layer Error: Cannot remove layers with multi-bottom and "
             "multi-top: "
          << cur_layer.name();
      DLOG(INFO) << "Remove " << type << " Type Layer: " << cur_layer.name();
      if (cur_layer.bottom_size() == 0 || cur_layer.top_size() == 0) {
        continue;
      }
      for (size_t j = 0; j < model_in.layer(i).top_size(); j++) {
        type_blobs.insert(
            make_pair(model_in.layer(i).top(j), model_in.layer(i).bottom(0)));
      }
    } else {
      LayerParameter *layer = model_out.add_layer();
      *layer = model_in.layer(i);
      for (size_t i = 0; i < layer->bottom_size(); i++) {
        auto res = type_blobs.find(layer->bottom(i));
        if (res != type_blobs.end()) {
          layer->set_bottom(i, res->second);
        }
      }
      for (size_t i = 0; i < layer->top_size(); i++) {
        auto res = type_blobs.find(layer->top(i));
        if (res != type_blobs.end()) {
          layer->set_top(i, res->second);
        }
      }
    }
  }

  return model_out;
}

NetParameter RemoveLayerByPhase(const NetParameter &model_in,
                                caffe::Phase phase) {
  NetParameter model_out;
  string phase_string = phase == caffe::TRAIN ? "TRAIN" : "TEST";
  map<string, string> phase_blobs;

  for (size_t i = 0; i < model_in.layer_size(); i++) {
    auto &cur_layer = model_in.layer(i);

    if (cur_layer.include_size() == 1 &&
        cur_layer.include(0).phase() == phase) {
      CHECK(!(cur_layer.bottom_size() > 1 && cur_layer.top_size() > 1))
          << "Remove Layer Error: Cannot remove layers with multi-bottom and "
             "multi-top: "
          << cur_layer.name();
      DLOG(INFO) << "Remove " << phase_string
                 << " Phase Layer: " << cur_layer.name();
      if (cur_layer.bottom_size() == 0 || cur_layer.top_size() == 0) {
        continue;
      }
      if (cur_layer.bottom_size() > 1) {
        for (size_t j = 0; j < model_in.layer(i).bottom_size(); j++) {
          phase_blobs.insert(
              make_pair(model_in.layer(i).bottom(j), model_in.layer(i).top(0)));
        }
      } else if (cur_layer.top_size() > 1) {
        for (size_t j = 0; j < model_in.layer(i).top_size(); j++) {
          phase_blobs.insert(
              make_pair(model_in.layer(i).top(j), model_in.layer(i).bottom(0)));
        }
      }
    } else {
      LayerParameter *layer = model_out.add_layer();
      *layer = model_in.layer(i);
      for (size_t i = 0; i < layer->bottom_size(); i++) {
        auto res = phase_blobs.find(layer->bottom(i));
        if (res != phase_blobs.end()) {
          layer->set_bottom(i, res->second);
        }
      }
      for (size_t i = 0; i < layer->top_size(); i++) {
        auto res = phase_blobs.find(layer->top(i));
        if (res != phase_blobs.end()) {
          layer->set_top(i, res->second);
        }
      }
    }
  }

  return model_out;
}

NetParameter RemoveLayerByName(const NetParameter &model_in,
                               const string name) {
  NetParameter model_out;
  map<string, string> name_blobs;

  for (size_t i = 0; i < model_in.layer_size(); i++) {
    auto &cur_layer = model_in.layer(i);
    if (cur_layer.name() == name) {
      CHECK(!(cur_layer.bottom_size() > 1 && cur_layer.top_size() > 1))
          << "Remove Layer Error: Cannot remove layers with multi-bottom and "
             "multi-top: "
          << cur_layer.name();
      DLOG(INFO) << "Remove " << name << " Name Layer: " << cur_layer.name();
      if (cur_layer.bottom_size() > 1) {
        for (size_t j = 0; j < model_in.layer(i).bottom_size(); j++) {
          name_blobs.insert(
              make_pair(model_in.layer(i).bottom(j), model_in.layer(i).top(0)));
        }
      } else if (cur_layer.top_size() > 1) {
        for (size_t j = 0; j < model_in.layer(i).top_size(); j++) {
          name_blobs.insert(
              make_pair(model_in.layer(i).top(j), model_in.layer(i).bottom(0)));
        }
      }
    } else {
      LayerParameter *layer = model_out.add_layer();
      *layer = model_in.layer(i);
      for (size_t i = 0; i < layer->bottom_size(); i++) {
        auto res = name_blobs.find(layer->bottom(i));
        if (res != name_blobs.end()) {
          layer->set_bottom(i, res->second);
        }
      }
      for (size_t i = 0; i < layer->top_size(); i++) {
        auto res = name_blobs.find(layer->top(i));
        if (res != name_blobs.end()) {
          layer->set_top(i, res->second);
        }
      }
    }
  }

  return model_out;
}

bool CheckImageDataLayer(const LayerParameter *data_layer) {
  CHECK_EQ(data_layer->type(), "ImageData");
  const int new_height = data_layer->image_data_param().new_height();
  const int new_width = data_layer->image_data_param().new_width();
  const bool is_color = data_layer->image_data_param().is_color();
  string root_folder = data_layer->image_data_param().root_folder();

  CHECK((new_height == 0 && new_width == 0) ||
        (new_height > 0 && new_width > 0))
      << " Please set new_height and new_width at the same time in "
         "ImageDataLayer.";
  // Read the file with filenames and labels
  const string &source = data_layer->image_data_param().source();
  LOG(INFO) << "Opening file " << source;
  std::ifstream infile(source.c_str());
  string filename;
  int label;
  CHECK(infile.is_open())
      << " Could not open file: " << source
      << ", please check the root_folder and source in ImageDataLayer.";
  vector<std::pair<std::string, int>> lines_;
  while (infile >> filename >> label) {
    lines_.push_back(std::make_pair(filename, label));
  }
  LOG(INFO) << "A total of " << lines_.size() << " images.";

  int lines_id_ = 0;
  // Read an image, and use it to initialize the top blob.
  cv::Mat cv_img = ReadImageToCVMat(root_folder + lines_[lines_id_].first,
                                    new_height, new_width, is_color);
  CHECK(cv_img.data) << " Could not load " << lines_[lines_id_].first
                     << ", please check the image file.";

  return true;
}

bool CheckModel(const NetParameter &net_param) {
  bool model_check = true;
  ostringstream ss;
  ss << "\n";
  for (auto i = 0; i < net_param.layer_size(); i++) {
    if (net_param.layer(i).type() == "ImageData") {
      CheckImageDataLayer(&net_param.layer(i));
    } else if (net_param.layer(i).type() == "LRN") {
      ss << "*** Deephi DNNDK could not support LRN Layer, ("
         << net_param.layer(i).name()
         << "), please delete it or replace it with BatchNorm + Scale Layer "
            "and "
            "retrain.\n";
      model_check = false;
    }
  }
  CHECK(model_check) << ss.str();
  return model_check;
}

/*
 *  white_list_entry:
 *  <support, comment>
 *  support:
 *     0. do not support
 *     1. do support
 *     2. will be processed by tool-chain
 *     3. ignore
*/
typedef std::tuple<int, string> white_list_entry;
typedef std::map<string, white_list_entry> white_list;

bool CheckModelAndSave(const NetParameter &net_param, const string filename) {
  bool pass = true;

  white_list wl;
  // DPU do not support
  wl.emplace("LRN",
             make_tuple(0, "Suggest: Replace it with BatchNorm+Scale Layer"));
  wl.emplace("Normalize",
             make_tuple(0, "Suggest: Replace it with BatchNorm+Scale Layer"));
  wl.emplace("Tiling", make_tuple(0, ""));
  wl.emplace("Softmax", make_tuple(0, ""));
  wl.emplace("Upsample", make_tuple(0, ""));
  // DPU do support
  wl.emplace("Convolution", make_tuple(1, ""));
  wl.emplace("Deconvolution", make_tuple(1, ""));
  wl.emplace("InnerProduct", make_tuple(1, ""));
  wl.emplace("Eltwise", make_tuple(1, ""));
  wl.emplace("Pooling", make_tuple(1, ""));
  wl.emplace("ReLU", make_tuple(1, ""));
  wl.emplace("PReLU", make_tuple(1, ""));
  wl.emplace("Concat", make_tuple(1, ""));
  // Will be processed by toolchain
  wl.emplace("BatchNorm", make_tuple(2, "Will be merged into Convolution"));
  wl.emplace("Scale", make_tuple(2, "Will be merged into Convolution"));
  wl.emplace("SoftmaxWithLoss", make_tuple(2, "Will be convert to Softmax"));
  wl.emplace("ImageData", make_tuple(2, "Will be convert to Input"));
  wl.emplace("Data", make_tuple(2, "Will be convert to Input"));
  // Ignore
  wl.emplace("Flatten", make_tuple(3, ""));
  wl.emplace("Permute", make_tuple(3, ""));
  wl.emplace("PriorBox", make_tuple(3, ""));
  wl.emplace("Loss", make_tuple(3, ""));
  wl.emplace("Sigmoid", make_tuple(3, ""));
  wl.emplace("Slice", make_tuple(3, ""));
  wl.emplace("Dropout", make_tuple(3, ""));
  wl.emplace("Accuracy", make_tuple(3, ""));

  ofstream fs;
  fs.open(filename);
  for (auto i = 0; i < net_param.layer_size(); i++) {
    auto it = wl.find(net_param.layer(i).type());
    if (it != wl.end()) {
      auto &entry = it->second;
      auto &support = get<0>(entry);
      auto &comment = get<1>(entry);
      if (support == 0) {
        cout << i << ". " << net_param.layer(i).name() << " ("
             << net_param.layer(i).type() << ") do not support by Deephi DPU. "
             << comment << "\n";
        fs << i << " " << net_param.layer(i).name() << " "
           << net_param.layer(i).type() << " " << support << " \"" << comment
           << "\"\n";
        pass = false;
      } else if (support == 2) {
        // cout << i << ". " << net_param.layer(i).name() << " (" <<
        // net_param.layer(i).type()
        // << ") will be processed by toolchain. "
        // << comment << "\n";
        fs << i << " " << net_param.layer(i).name() << " "
           << net_param.layer(i).type() << " " << support << " \"" << comment
           << "\"\n";
      } else if (support == 3) {
        // cout << i << ". " << net_param.layer(i).name() << " (" <<
        // net_param.layer(i).type()
        // << ") can be ignored in forward. "
        // << comment << "\n";
        fs << i << " " << net_param.layer(i).name() << " "
           << net_param.layer(i).type() << " " << support << " \"" << comment
           << "\"\n";
      }
    } else {
      cout << i << ". " << net_param.layer(i).name() << " ("
           << net_param.layer(i).type()
           << ") is not known by DNNDK and may raise some error, please "
              "concat Deephi.\n";
    }
  }

  fs.close();
  return pass;
}

NetParameter RemoveBlobs(const NetParameter &model_in) {
  NetParameter model_out = model_in;
  for (size_t i = 0; i < model_out.layer_size(); i++) {
    model_out.mutable_layer(i)->clear_blobs();
  }
  return model_out;
}

template <typename Dtype>
void DataStat(const int n, const Dtype *data, const char *title) {
  double max = FLT_MIN, min = FLT_MAX;
  vector<int> count(11, 0);
  for (int i = 0; i < n; ++i) {
    double val = fabs(data[i]);
    min = fabs(val) < fabs(min) ? fabs(val) : min;
    max = fabs(val) > fabs(max) ? fabs(val) : max;
    if (val < 0.03125)
      ++count[0];
    else if (val < 0.0625)
      ++count[1];
    else if (val < 0.125)
      ++count[2];
    else if (val < 0.25)
      ++count[3];
    else if (val < 0.5)
      ++count[4];
    else if (val < 1)
      ++count[5];
    else if (val < 2)
      ++count[6];
    else if (val < 4)
      ++count[7];
    else if (val < 8)
      ++count[8];
    else if (val < 16)
      ++count[9];
    else
      ++count[10];
  }
  printf("%s max/min: %-9g %-9g\n", title, max, min);
  printf("Range    <2^-5    <2^-4    <2^-3    <0.25     <0.5       <1       <2 "
         "      <4       <8      <16     <max\n");
  printf("Count");
  for (int j = 0; j < count.size(); ++j)
    printf(" %8d", count[j]);
  printf("\n\n");
  fflush(stdout);
}

template <typename Dtype>
void BindWeightWithProfile(Blob<float> &new_weight, const Dtype *weight,
                           const Dtype *scale, const Dtype *var, float eps) {
#if defined(USE_CUDNN) and !defined(CPU_ONLY)
  eps = max(eps, float(CUDNN_BN_MIN_EPSILON));
#endif
  const int n = new_weight.count();
  const int kernel_dim = new_weight.count(1);
  auto bind_weight = new_weight.mutable_cpu_data();

//#define DEBUG_BN_MERGE
#ifdef DEBUG_BN_MERGE
  DataStat(n, weight, "Weights before merge BN");
#endif // DEBUG_BN_MERGE

  int m = n / kernel_dim;
  vector<double> factor(m, 1.0);
#ifdef DEBUG_BN_MERGE
  vector<double> inv_var(m, 1.0);
#endif // DEBUG_BN_MERGE
  for (int i = 0; i < m; i++) {
#ifdef DEBUG_BN_MERGE
    inv_var[i] = 1.0 / sqrt(var[i] + eps);
#endif // DEBUG_BN_MERGE
    factor[i] = double(scale[i]) / sqrt(double(var[i]) + eps);
  }

#ifdef DEBUG_BN_MERGE
  DataStat(m, scale, "Scale from BN+scale");
  DataStat(m, var, "Variance from BN+scale");
  DataStat<double>(m, &(inv_var[0]), "1/sqrt(var) from BN+scale");
  DataStat<double>(m, &(factor[0]), "Scale/sqrt(var) from BN+scale");
#endif // DEBUG_BN_MERGE

  for (int i = 0; i < n; i++) {
    int c = i / kernel_dim;
    bind_weight[i] = weight[i] * factor[c];
  }

#ifdef DEBUG_BN_MERGE
  DataStat(n, bind_weight, "Weights after merge BN");
#endif // DEBUG_BN_MERGE
}
template <typename Dtype>
void BindWeight(const int n, const int kernel_dim, const Dtype *weight,
                const Dtype *scale, const Dtype *var, Dtype *bind_weight,
                float eps) {
#if defined(USE_CUDNN) and !defined(CPU_ONLY)
  eps = max(eps, float(CUDNN_BN_MIN_EPSILON));
#endif

  int m = n / kernel_dim;
  vector<Dtype> factor(m, 1.0);
  for (int i = 0; i < m; i++)
    factor[i] = scale[i] / sqrt(var[i] + eps);

  for (int i = 0; i < n; i++) {
    int c = i / kernel_dim;
    bind_weight[i] = weight[i] * factor[c];
  }
}

template <typename Dtype>
void BindBias(const int n, const Dtype *mean, const Dtype *scale,
              const Dtype *var, const Dtype *bias_conv, const Dtype *bias_bn,
              Dtype *bind_bias, float eps) {
#if defined(USE_CUDNN) and !defined(CPU_ONLY)
  eps = max(eps, float(CUDNN_BN_MIN_EPSILON));
#endif

  if (bias_conv != nullptr) {
#ifdef DEBUG_BN_MERGE
    DataStat(n, bias_conv, "Bias before merge BN");
#endif // DEBUG_BN_MERGE
  }

  for (int i = 0; i < n; i++) {
    if (bias_conv != nullptr) {
      bind_bias[i] =
          Dtype(double(bias_bn[i]) +
                (double(bias_conv[i]) - double(mean[i])) * double(scale[i]) *
                    (1 / sqrt(double(var[i]) + eps)));
    } else {
      bind_bias[i] = Dtype(double(bias_bn[i]) -
                           double(mean[i]) * double(scale[i]) *
                               (1 / sqrt(double(var[i]) + eps)));
    }
  }

#ifdef DEBUG_BN_MERGE
  DataStat(n, bind_bias, "Bias after merge BN");
#endif // DEBUG_BN_MERGE
}

// BindBNConvWeight: Bind weight in merging BatchNorm + Conv to Conv
template <typename Dtype>
void BindBNConvWeight(const int num, const int ch, const int kernel_dim,
                      const Dtype *weight, const Dtype *scale, const Dtype *var,
                      Dtype *bind_weight, float eps) {
#if defined(USE_CUDNN) and !defined(CPU_ONLY)
  eps = max(eps, float(CUDNN_BN_MIN_EPSILON));
#endif
  int inner_kernel_dim = kernel_dim / ch;
  for (int i = 0; i < num; i++) {
    for (int j = 0; j < kernel_dim; j++) {
      int c = j / inner_kernel_dim;
      bind_weight[i * kernel_dim + j] =
          weight[i * kernel_dim + j] * scale[c] * (1 / sqrt(var[c] + eps));
    }
  }
}

// BindBNConvBias: Bind new bias in merging BatchNorm + Conv to Conv
template <typename Dtype>
void BindBNConvBias(const int num, const int ch, const int kernel_dim,
                    const Dtype *mean, const Dtype *scale, const Dtype *var,
                    const Dtype *weight, const Dtype *bias_conv,
                    const Dtype *bias_bn, Dtype *bind_bias, float eps) {
#if defined(USE_CUDNN) and !defined(CPU_ONLY)
  eps = max(eps, float(CUDNN_BN_MIN_EPSILON));
#endif
  for (int i = 0; i < num; i++) {
    bind_bias[i] = 0;
    for (int j = 0; j < ch; j++) {
      Dtype weight_sum = 0;
      for (int k = 0; k < kernel_dim / ch; k++) {
        weight_sum += weight[i * kernel_dim + j * kernel_dim / ch + k];
      }
      bind_bias[i] +=
          weight_sum *
          (bias_bn[j] - scale[j] * mean[j] * (1 / sqrt(var[j] + eps)));
    }
    if (bias_conv != nullptr) {
      bind_bias[i] += bias_conv[i];
    }
  }
}

template <typename Dtype>
void ScaleInvVar(const int n, const Dtype *scale, const Dtype *var,
                 Dtype *scale_inv_var, float eps) {
#if defined(USE_CUDNN) and !defined(CPU_ONLY)
  eps = max(eps, float(CUDNN_BN_MIN_EPSILON));
#endif
  for (int i = 0; i < n; i++) {
    scale_inv_var[i] = scale[i] * (1 / sqrt(var[i] + eps));
  }
}

vector<vector<bool>> BuildConnections(const NetParameter &net) {
  vector<vector<bool>> connect(net.layer_size(),
                               vector<bool>(net.layer_size(), false));

  for (size_t source_layer_index = 0; source_layer_index < net.layer_size() - 1;
       source_layer_index++) {
    for (size_t top_index = 0;
         top_index < net.layer(source_layer_index).top_size(); top_index++) {
      for (size_t target_layer_index = source_layer_index + 1;
           target_layer_index < net.layer_size(); target_layer_index++) {
        for (size_t bottom_index = 0;
             bottom_index < net.layer(target_layer_index).bottom_size();
             bottom_index++) {
          if (net.layer(source_layer_index).top(top_index) ==
              net.layer(target_layer_index).bottom(bottom_index)) {
            connect[source_layer_index][target_layer_index] = true;
            connect[target_layer_index][source_layer_index] = true;
          }
        }
      }
    }
  }
  return connect;
}

void ShowConnections(NetParameter net, vector<vector<bool>> connect) {
  for (int i = 0; i < connect.size(); i++) {
    for (int j = 0; j < connect[i].size(); j++) {
      if (connect[i][j] == true) {
        std::cout << "layer[" << i << "](" << net.layer(i).name() << ")("
                  << net.layer(i).type() << ")"
                  << " <----> "
                  << "layer[" << j << "](" << net.layer(j).name() << ")("
                  << net.layer(j).type() << ")" << std::endl;
      }
    }
    std::cout << std::endl;
  }
}

// Infer net type from layer types, will be override by FLAGS_net_type. Mainly
// used for net testing.
// Options: detection, segmentation, other
const string InferNetType(const NetParameter &net_param) {
  for (auto i = 0; i < net_param.layer_size(); i++) {
    if (net_param.layer(i).type() == "AnnotatedData") {
      return "detection";
    } else if (net_param.layer(i).type() == "SegmentPixelIOU") {
      return "segmentation";
    } else if (net_param.layer(i).type() == "YoloEvalDetection") {
      return "detection";
    }
  }
  return "other";
}

const vector<int> GetInputShape(NetParameter net_param,
                                const string &input_blob) {
  net_param.mutable_state()->set_phase(TRAIN);
  Net<float> net(net_param);
  CHECK(net.has_blob(input_blob))
      << " No blob named " << input_blob
      << " found, Please rename model's input blob name as " << input_blob;
  return net.blob_by_name(input_blob)->shape();
}

TransformationParameter GetTransformParam(const NetParameter &net) {
  TransformationParameter transform_param;
  for (int i = 0; i < net.layer_size(); i++) {
    const LayerParameter &cur_layer = net.layer(i);
    if (cur_layer.has_transform_param()) {
      transform_param = cur_layer.transform_param();
    }
  }
  return transform_param;
}

int GetNextLayer(const NetParameter &net, const int index, const string type,
                 const vector<vector<bool>> &connect) {
  for (int i = index + 1; i < net.layer_size(); i++) {
    if (connect[i][index] && net.layer(i).type() == type) {
      return i;
    }
  }
  return -1;
}

int GetPreviousLayer(const NetParameter &net, const int index,
                     const string type, const vector<vector<bool>> &connect) {
  for (int i = index - 1; i >= 0; i--) {
    if (connect[i][index] && net.layer(i).type() == type) {
      return i;
    }
  }
  return -1;
}

bool IsInplace(const LayerParameter *layer) {
  if (layer->bottom_size() == 1 && layer->top_size() == 1) {
    if (layer->bottom(0) == layer->top(0)) {
      return true;
    } else {
      return false;
    }
  } else {
    return false;
  }
}

int GetLastInputLayer(const NetParameter &net) {
  // find last input layer
  for (int i = net.layer_size() - 1; i >= 0; i--) {
    for (int j = 0; j < net.layer(i).top_size(); j++) {
      if (net.layer(i).top(j) == "data") {
        return i;
      }
    }
  }
  for (int i = net.layer_size() - 1; i >= 0; i--) {
    if (net.layer(i).type() == "Data" || net.layer(i).type() == "ImageData" ||
        net.layer(i).type() == "BoxData" ||
        net.layer(i).type() == "EnhancedImageData" ||
        net.layer(i).type() == "Input" || net.layer(i).type() == "Power" ||
        net.layer(i).type() == "DetectNetTransformation" ||
        net.layer(i).type() == "AnnotatedData") {
      return i;
    }
  }
  return -1;
}

// Merge model
void MergeConvBatchNorm2Conv(const NetParameter &net_in, NetParameter &net_out,
                             const int conv_index, const int bn_index,
                             vector<bool> &processed, bool binary) {
  const LayerParameter &conv_param = net_in.layer(conv_index);
  const LayerParameter &bn_param = net_in.layer(bn_index);
  LayerParameter *layer_param = net_out.add_layer();
  layer_param->CopyFrom(conv_param);
  layer_param->set_top(0, bn_param.top(0));
  layer_param->mutable_convolution_param()->set_bias_term(true);
  // Set Param
  layer_param->clear_param();
  float lr_mult[2] = {1, 1};
  float decay_mult[2] = {1, 0};
  for (int j = 0; j < 2; j++) {
    ParamSpec *param = layer_param->add_param();
    param->set_lr_mult(lr_mult[j]);
    param->set_decay_mult(decay_mult[j]);
  }
  layer_param->clear_batch_norm_param();
  // Copy Blobs
  if (binary) {
    CHECK_EQ(bn_param.blobs_size(), 5)
        << " Wrong BatchNormLayer blob size of layer: " << bn_param.name();
    layer_param->clear_blobs();
    auto weight = conv_param.blobs(0).data().data();
    const float *bias_conv = nullptr;
    if (conv_param.convolution_param().bias_term()) {
      bias_conv = conv_param.blobs(1).data().data();
    }
    auto scale = bn_param.blobs(0).data().data();
    auto bias_bn = bn_param.blobs(1).data().data();
    auto mean = bn_param.blobs(2).data().data();
    auto var = bn_param.blobs(3).data().data();

    Blob<float> new_weight;
    Blob<float> new_bias;
    new_weight.FromProto(conv_param.blobs(0), 1);
    new_bias.FromProto(bn_param.blobs(1), 1);

    // BindWeight(new_weight.count(), new_weight.count(1), weight, scale, var,
    //           new_weight.mutable_cpu_data());
    float eps = bn_param.batch_norm_param().eps();
    BindWeightWithProfile(new_weight, weight, scale, var, eps);

    if (conv_param.convolution_param().bias_term()) {
      BindBias(new_bias.count(), mean, scale, var, bias_conv, bias_bn,
               new_bias.mutable_cpu_data(), eps);
    } else {
      BindBias(new_bias.count(), mean, scale, var, (float *)nullptr, bias_bn,
               new_bias.mutable_cpu_data(), eps);
    }
    new_weight.ToProto(layer_param->add_blobs());
    new_bias.Reshape({new_bias.count()});
    new_bias.ToProto(layer_param->add_blobs());
  }
  processed[conv_index] = true;
  processed[bn_index] = true;
}

void MergeFCBatchNorm2FC(const NetParameter &net_in, NetParameter &net_out,
                         const int fc_index, const int bn_index,
                         vector<bool> &processed, bool binary) {
  const LayerParameter &fc_param = net_in.layer(fc_index);
  const LayerParameter &bn_param = net_in.layer(bn_index);
  LayerParameter *layer_param = net_out.add_layer();
  layer_param->CopyFrom(fc_param);
  layer_param->set_top(0, bn_param.top(0));
  layer_param->mutable_inner_product_param()->set_bias_term(true);
  // Set Param
  layer_param->clear_param();
  float lr_mult[2] = {1, 1};
  float decay_mult[2] = {1, 0};
  for (int j = 0; j < 2; j++) {
    ParamSpec *param = layer_param->add_param();
    param->set_lr_mult(lr_mult[j]);
    param->set_decay_mult(decay_mult[j]);
  }
  layer_param->clear_batch_norm_param();
  // Copy Blobs
  if (binary) {
    CHECK_EQ(bn_param.blobs_size(), 5)
        << " Wrong BatchNormLayer blob size of layer: " << bn_param.name();
    layer_param->clear_blobs();
    auto weight = fc_param.blobs(0).data().data();
    const float *bias_fc = nullptr;
    if (fc_param.inner_product_param().bias_term()) {
      bias_fc = fc_param.blobs(1).data().data();
    }
    auto scale = bn_param.blobs(0).data().data();
    auto bias_bn = bn_param.blobs(1).data().data();
    auto mean = bn_param.blobs(2).data().data();
    auto var = bn_param.blobs(3).data().data();

    Blob<float> new_weight;
    Blob<float> new_bias;
    new_weight.FromProto(fc_param.blobs(0), 1);
    new_bias.FromProto(bn_param.blobs(1), 1);

    float eps = bn_param.batch_norm_param().eps();
    BindWeight(new_weight.count(), new_weight.count(1), weight, scale, var,
               new_weight.mutable_cpu_data(), eps);
    if (fc_param.inner_product_param().bias_term()) {
      BindBias(new_bias.count(), mean, scale, var, bias_fc, bias_bn,
               new_bias.mutable_cpu_data(), eps);
    } else {
      BindBias(new_bias.count(), mean, scale, var, (float *)nullptr, bias_bn,
               new_bias.mutable_cpu_data(), eps);
    }
    new_weight.ToProto(layer_param->add_blobs());
    new_bias.Reshape({new_bias.count()});
    new_bias.ToProto(layer_param->add_blobs());
  }
  processed[fc_index] = true;
  processed[bn_index] = true;
}

// Merge model
void MergeBatchNormConv2Conv(const NetParameter &net_in, NetParameter &net_out,
                             const int bn_index, const int conv_index,
                             vector<bool> &processed, bool binary) {
  const LayerParameter &conv_param = net_in.layer(conv_index);
  const LayerParameter &bn_param = net_in.layer(bn_index);
  LayerParameter *layer_param = net_out.add_layer();
  layer_param->CopyFrom(conv_param);
  layer_param->set_bottom(0, bn_param.bottom(0));
  layer_param->mutable_convolution_param()->set_bias_term(true);
  // Set Param
  layer_param->clear_param();
  float lr_mult[2] = {1, 1};
  float decay_mult[2] = {1, 0};
  for (int j = 0; j < 2; j++) {
    ParamSpec *param = layer_param->add_param();
    param->set_lr_mult(lr_mult[j]);
    param->set_decay_mult(decay_mult[j]);
  }
  layer_param->clear_batch_norm_param();
  // Copy Blobs
  if (binary) {
    CHECK_EQ(bn_param.blobs_size(), 5)
        << " Wrong BatchNormLayer blob size of layer: " << bn_param.name();
    layer_param->clear_blobs();
    auto weight = conv_param.blobs(0).data().data();
    const float *bias_conv = nullptr;
    if (conv_param.convolution_param().bias_term()) {
      bias_conv = conv_param.blobs(1).data().data();
    }
    auto scale = bn_param.blobs(0).data().data();
    auto bias_bn = bn_param.blobs(1).data().data();
    auto mean = bn_param.blobs(2).data().data();
    auto var = bn_param.blobs(3).data().data();

    Blob<float> new_weight;
    Blob<float> new_bias;
    new_weight.FromProto(conv_param.blobs(0), 1);
    if (conv_param.convolution_param().bias_term()) {
      new_bias.FromProto(conv_param.blobs(1), 1);
    } else {
      new_bias.Reshape({1, new_weight.shape(1), 1, 1});
      LOG(INFO) << "new_bias.shape(): " << new_bias.shape_string();
    }

    float eps = bn_param.batch_norm_param().eps();
    BindBNConvWeight(new_weight.num(), new_weight.channels(),
                     new_weight.count(1), weight, scale, var,
                     new_weight.mutable_cpu_data(), eps);
    if (conv_param.convolution_param().bias_term()) {
      BindBNConvBias(new_weight.num(), new_weight.channels(),
                     new_weight.count(1), mean, scale, var, weight, bias_conv,
                     bias_bn, new_bias.mutable_cpu_data(), eps);
    } else {
      BindBNConvBias(new_weight.num(), new_weight.channels(),
                     new_weight.count(1), mean, scale, var, weight,
                     (float *)nullptr, bias_bn, new_bias.mutable_cpu_data(),
                     eps);
    }
    new_weight.ToProto(layer_param->add_blobs());
    new_bias.Reshape({new_bias.count()});
    new_bias.ToProto(layer_param->add_blobs());
  }
  processed[conv_index] = true;
  processed[bn_index] = true;
}

void MergeBatchNorm2Scale(const NetParameter &net_in, NetParameter &net_out,
                          const int bn_index, vector<bool> &processed,
                          bool binary) {
  const LayerParameter &bn_param = net_in.layer(bn_index);
  LayerParameter *layer_param = net_out.add_layer();
  layer_param->CopyFrom(bn_param);
  layer_param->set_type("Scale");
  layer_param->mutable_scale_param()->set_bias_term(true);
  // Set Param
  layer_param->clear_param();
  float lr_mult[2] = {1, 1};
  float decay_mult[2] = {1, 0};
  for (int j = 0; j < 2; j++) {
    ParamSpec *param = layer_param->add_param();
    param->set_lr_mult(lr_mult[j]);
    param->set_decay_mult(decay_mult[j]);
  }
  layer_param->clear_batch_norm_param();

  if (binary) {
    CHECK_EQ(bn_param.blobs_size(), 5)
        << " Wrong BatchNormLayer blob size of layer: " << bn_param.name();
    layer_param->clear_blobs();

    Blob<float> new_weight;
    Blob<float> new_bias;
    new_weight.FromProto(bn_param.blobs(0), 1);
    new_bias.FromProto(bn_param.blobs(1), 1);

    auto scale = bn_param.blobs(0).data().data();
    auto bn_bias = bn_param.blobs(1).data().data();
    auto mean = bn_param.blobs(2).data().data();
    auto var = bn_param.blobs(3).data().data();

    // Bind scale and bias
    float eps = bn_param.batch_norm_param().eps();
    ScaleInvVar(new_weight.channels(), scale, var,
                new_weight.mutable_cpu_data(), eps);
    BindBias(new_bias.channels(), mean, scale, var, (float *)nullptr, bn_bias,
             new_bias.mutable_cpu_data(), eps);

    new_weight.Reshape({new_weight.count()});
    new_weight.ToProto(layer_param->add_blobs());
    new_bias.Reshape({new_bias.count()});
    new_bias.ToProto(layer_param->add_blobs());
  }
  processed[bn_index] = true;
}

void MergeBvlcBatchNormScale2BatchNorm(const NetParameter &net_in,
                                       NetParameter &net_out,
                                       const int bn_index, const int sc_index,
                                       vector<bool> &processed, bool binary) {
  const LayerParameter &bn_param = net_in.layer(bn_index);
  const LayerParameter &sc_param = net_in.layer(sc_index);
  LayerParameter *layer_param = net_out.add_layer();
  layer_param->CopyFrom(bn_param);
  layer_param->set_top(0, sc_param.top(0));
  // Set Param
  layer_param->clear_param();
  float lr_mult[4] = {1, 1, 0, 0};
  float decay_mult[4] = {0, 0, 0, 0};
  for (int j = 0; j < 4; j++) {
    ParamSpec *param = layer_param->add_param();
    param->set_lr_mult(lr_mult[j]);
    param->set_decay_mult(decay_mult[j]);
  }
  // Set batch_norm_param
  if (sc_param.scale_param().has_filler()) {
    layer_param->mutable_batch_norm_param()->mutable_scale_filler()->CopyFrom(
        sc_param.scale_param().filler());
  } else {
    layer_param->mutable_batch_norm_param()->mutable_scale_filler()->set_type(
        "constant");
    layer_param->mutable_batch_norm_param()->mutable_scale_filler()->set_value(
        1.);
  }
  if (sc_param.scale_param().has_bias_filler()) {
    layer_param->mutable_batch_norm_param()->mutable_bias_filler()->CopyFrom(
        sc_param.scale_param().bias_filler());
  } else {
    layer_param->mutable_batch_norm_param()->mutable_bias_filler()->set_type(
        "constant");
    layer_param->mutable_batch_norm_param()->mutable_bias_filler()->set_value(
        0.);
  }
  // Copy Blobs
  if (binary) {
    CHECK_EQ(sc_param.blobs_size(), 2)
        << " Wrong ScaleLayer blob size of layer: " << sc_param.name();
    CHECK_EQ(bn_param.blobs_size(), 3)
        << " Wrong BatchNormLayer blob size of layer: " << bn_param.name();
    layer_param->clear_blobs();
    *(layer_param->add_blobs()) = sc_param.blobs(0); // scale
    *(layer_param->add_blobs()) = sc_param.blobs(1); // bias
    *(layer_param->add_blobs()) = bn_param.blobs(0); // mean
    *(layer_param->add_blobs()) = bn_param.blobs(1); // variance
    *(layer_param->add_blobs()) = bn_param.blobs(2); // redundant_blob
    caffe_scal<float>(
        layer_param->blobs(0).data_size(),
        1 / layer_param->blobs(4).data().data()[0],
        layer_param->mutable_blobs(2)->mutable_data()->mutable_data());
    caffe_scal<float>(
        layer_param->blobs(0).data_size(),
        1 / layer_param->blobs(4).data().data()[0],
        layer_param->mutable_blobs(3)->mutable_data()->mutable_data());
    layer_param->mutable_blobs(4)->set_data(0, 1);
    // Change blob dim
    for (size_t i = 0; i < layer_param->blobs_size() - 1; i++) {
      if (layer_param->blobs(i).shape().dim_size() == 1) {
        DLOG(INFO) << "Update blob shape dim: " << layer_param->name()
                   << " blob[" << i << "]";
        layer_param->mutable_blobs(i)->mutable_shape()->clear_dim();
        layer_param->mutable_blobs(i)->mutable_shape()->add_dim(1);
        layer_param->mutable_blobs(i)->mutable_shape()->add_dim(
            bn_param.blobs(0).shape().dim(0));
        layer_param->mutable_blobs(i)->mutable_shape()->add_dim(1);
        layer_param->mutable_blobs(i)->mutable_shape()->add_dim(1);
      } else if (layer_param->blobs(i).shape().dim_size() != 4) {
        DLOG(WARNING) << "Warning: wrong dim_size for layer: "
                      << layer_param->name() << " blob[" << i << "]";
      }
    }
  }
  processed[sc_index] = true;
  processed[bn_index] = true;
}

void ConvertBvlcBatchNorm(const NetParameter &net_in, NetParameter &net_out,
                          const int bn_index, vector<bool> &processed,
                          bool binary) {
  const LayerParameter &bn_param = net_in.layer(bn_index);
  LayerParameter *layer_param = net_out.add_layer();
  layer_param->CopyFrom(bn_param);
  layer_param->set_type("BatchNorm");
  DLOG(INFO) << "Update BVLC BatchNorm: " << layer_param->name();
  // Copy Blobs
  if (binary) {
    CHECK_EQ(bn_param.blobs_size(), 3)
        << "Wrong Bvlc BatchNormLayer blobs_size of layer: " << bn_param.name();
    layer_param->clear_blobs();
    // Make up scale and bias blobs
    *(layer_param->add_blobs()) = bn_param.blobs(0); // scale
    *(layer_param->add_blobs()) = bn_param.blobs(0); // bias
    for (size_t i = 0; i < bn_param.blobs(0).shape().dim(0); i++) {
      layer_param->mutable_blobs(0)->set_data(i, 1);
      layer_param->mutable_blobs(1)->set_data(i, 0);
    }
    // Copy mean and variance blobs, take care of scale_factor
    *(layer_param->add_blobs()) = bn_param.blobs(0); // mean
    *(layer_param->add_blobs()) = bn_param.blobs(1); // variance
    float scale_factor =
        bn_param.blobs(2).data(0) == 0 ? 0 : 1 / bn_param.blobs(2).data(0);
    for (size_t i = 0; i < bn_param.blobs(0).shape().dim(0); i++) {
      layer_param->mutable_blobs(2)->set_data(i, bn_param.blobs(0).data(i) *
                                                     scale_factor);
      layer_param->mutable_blobs(3)->set_data(i, bn_param.blobs(1).data(i) *
                                                     scale_factor);
    }
    BlobProto *redundant_blob = layer_param->add_blobs(); // redundant_blob
    redundant_blob->mutable_shape()->add_dim(1);
    redundant_blob->mutable_shape()->add_dim(1);
    redundant_blob->mutable_shape()->add_dim(1);
    redundant_blob->mutable_shape()->add_dim(1);
    redundant_blob->add_data(1);
    // Change blob dim
    for (size_t i = 0; i < layer_param->blobs_size() - 1; i++) {
      if (layer_param->blobs(i).shape().dim_size() == 1) {
        DLOG(INFO) << "Update blob shape dim: " << layer_param->name()
                   << " blob[" << i << "]";
        layer_param->mutable_blobs(i)->mutable_shape()->clear_dim();
        layer_param->mutable_blobs(i)->mutable_shape()->add_dim(1);
        layer_param->mutable_blobs(i)->mutable_shape()->add_dim(
            bn_param.blobs(0).shape().dim(0));
        layer_param->mutable_blobs(i)->mutable_shape()->add_dim(1);
        layer_param->mutable_blobs(i)->mutable_shape()->add_dim(1);
      } else if (layer_param->blobs(i).shape().dim_size() != 4) {
        DLOG(WARNING) << "Warning: wrong dim_size for layer: "
                      << layer_param->name() << " blob[" << i << "]";
      }
    }
  }
  processed[bn_index] = true;
}

void ConvertBN2BatchNorm(const NetParameter &net_in, NetParameter &net_out,
                         const int bn_index, vector<bool> &processed,
                         bool binary) {
  const LayerParameter &bn_param = net_in.layer(bn_index);
  LayerParameter *layer_param = net_out.add_layer();
  layer_param->CopyFrom(bn_param);
  layer_param->set_type("BatchNorm");
  DLOG(INFO) << "Change BN to BatchNorm: " << layer_param->name();
  // Copy Blobs
  if (binary) {
    CHECK_EQ(bn_param.blobs_size(), 4) << "Wrong BNLayer blobs_size of layer: "
                                       << bn_param.name();
    layer_param->clear_blobs();
    *(layer_param->add_blobs()) = bn_param.blobs(0);      // scale
    *(layer_param->add_blobs()) = bn_param.blobs(1);      // bias
    *(layer_param->add_blobs()) = bn_param.blobs(2);      // mean
    *(layer_param->add_blobs()) = bn_param.blobs(3);      // variance
    BlobProto *redundant_blob = layer_param->add_blobs(); // redundant_blob
    redundant_blob->mutable_shape()->add_dim(1);
    redundant_blob->mutable_shape()->add_dim(1);
    redundant_blob->mutable_shape()->add_dim(1);
    redundant_blob->mutable_shape()->add_dim(1);
    redundant_blob->add_data(1);
    // Change blob dim
    for (size_t i = 0; i < layer_param->blobs_size() - 1; i++) {
      if (layer_param->blobs(i).shape().dim_size() == 1) {
        DLOG(INFO) << "Update blob shape dim: " << layer_param->name()
                   << " blob[" << i << "]";
        layer_param->mutable_blobs(i)->mutable_shape()->clear_dim();
        layer_param->mutable_blobs(i)->mutable_shape()->add_dim(1);
        layer_param->mutable_blobs(i)->mutable_shape()->add_dim(
            bn_param.blobs(0).shape().dim(0));
        layer_param->mutable_blobs(i)->mutable_shape()->add_dim(1);
        layer_param->mutable_blobs(i)->mutable_shape()->add_dim(1);
      } else if (layer_param->blobs(i).shape().dim_size() != 4) {
        DLOG(WARNING) << "Warning: wrong dim_size for layer: "
                      << layer_param->name() << " blob[" << i << "]";
      }
    }
  }
  processed[bn_index] = true;
}

void ConvertSoftLoss2Soft(NetParameter &net, const int soft_index) {
  CHECK_EQ(net.layer(soft_index).type(), "SoftmaxWithLoss");
  LayerParameter *soft_param = net.mutable_layer(soft_index);
  soft_param->set_type("Softmax");
  soft_param->clear_include();
  CHECK_GE(soft_param->bottom_size(), 2);
  soft_param->mutable_bottom()->DeleteSubrange(1,
                                               soft_param->bottom_size() - 1);
  soft_param->mutable_top()->DeleteSubrange(1, soft_param->top_size() - 1);
}

// Convert train_val.prototxt to deploy.prototxt
// Function:
//      1) Remove Split
//      2) Remove Train Phase Layer
//      3) Remove Test Phase Layer
//      4) Remove Dropout Layer
//      5) Convert SoftmaxWithLoss to Softmax
NetParameter ConvertTrain2Deploy(const NetParameter &train_net) {
  int debugMode = 0;
  if (getenv("DECENT_DEBUG"))
    debugMode = stoi(getenv("DECENT_DEBUG"));

  NetParameter deploy_net = RemoveLayerByPhase(train_net, TRAIN);
  deploy_net = RemoveLayerByPhase(deploy_net, TEST);
  deploy_net = RemoveLayerByType(deploy_net, "Split");
  deploy_net = RemoveLayerByType(deploy_net, "Dropout");
  if (debugMode > 1)
    // debug mode to match Caffe deploy dev branch accuracy
    deploy_net = RemoveLayerByType(deploy_net, "SoftmaxWithLoss");
  else {
    for (auto i = 0; i < deploy_net.layer_size(); i++) {
      auto &cur_layer = deploy_net.layer(i);
      if (cur_layer.type() == "SoftmaxWithLoss") {
        ConvertSoftLoss2Soft(deploy_net, i);
      }
    }
  }
  return deploy_net;
}

NetParameter MergeBvlcBatchNormInNet(const NetParameter &net_in, bool binary) {
  vector<vector<bool>> connect = BuildConnections(net_in);
  vector<bool> processed(net_in.layer_size(), false);
  NetParameter net_out;

  for (int i = 0; i < net_in.layer_size(); i++) {
    if (processed[i]) {
      continue;
    }
    const LayerParameter *cur_layer = &net_in.layer(i);
    if (cur_layer->type() == "BatchNorm") {
      int scale_index = GetNextLayer(net_in, i, "Scale", connect);
      int b_size = cur_layer->blobs_size();
      if (binary)
        CHECK(b_size == 3 || b_size == 5)
            << "Wrong Blobs for BatchNorm: " << b_size
            << ", Only support BatchNorm with 3 blobs or 5 blobs";
      if (!binary) {
        // For prototxt
        if (scale_index != -1) {
          DLOG(INFO) << " Merge BvlcBatchNorm + Scale -> BatchNorm: "
                     << net_in.layer(i).name() << " + "
                     << net_in.layer(scale_index).name();
          MergeBvlcBatchNormScale2BatchNorm(net_in, net_out, i, scale_index,
                                            processed, binary);
        } else {
          *(net_out.add_layer()) = *cur_layer;
          processed[i] = true;
        }
      } else {
        // For Caffemodel
        if (scale_index != -1) {
          CHECK(b_size == 3)
              << "Wrong Blobs for BatchNorm Layer " << net_in.layer(i).name()
              << ", BatchNorm with 5 blobs should not followed by Scale layer.";
          DLOG(INFO) << " Merge BvlcBatchNorm + Scale -> BatchNorm: "
                     << net_in.layer(i).name() << " + "
                     << net_in.layer(scale_index).name();
          MergeBvlcBatchNormScale2BatchNorm(net_in, net_out, i, scale_index,
                                            processed, binary);
        } else {
          if (b_size == 3) {
            LOG(WARNING) << "BVLC BatchNormLayer without ScaleLayer detected: "
                         << i << ", it will be converted to NVCaffe Format.";
            ConvertBvlcBatchNorm(net_in, net_out, i, processed, binary);
          } else {
            *(net_out.add_layer()) = *cur_layer;
            processed[i] = true;
          }
        }
      }

    } else if (cur_layer->type() == "BN") {
      ConvertBN2BatchNorm(net_in, net_out, i, processed, binary);

    } else {
      *(net_out.add_layer()) = *cur_layer;
      processed[i] = true;
    }
  }
  return net_out;
}

NetParameter MergePostBatchNormInNet(const NetParameter &net_in, bool binary,
                                     bool keep_convbn) {
  vector<vector<bool>> connect = BuildConnections(net_in);
  vector<bool> processed(net_in.layer_size(), false);
  NetParameter net_out;

  for (int i = 0; i < net_in.layer_size(); i++) {
    if (processed[i]) {
      continue;
    }
    const LayerParameter *cur_layer = &net_in.layer(i);
    // Merge Convolution + BatchNorm
    if (cur_layer->type() == "Convolution") {
      int bn_index = GetNextLayer(net_in, i, "BatchNorm", connect);
      int relu_index = GetNextLayer(net_in, i, "ReLU", connect);
      // stop merge for Convolution + Relu + Batchnorm
      if (bn_index != -1 && (relu_index == -1 || relu_index > bn_index)) {
        if (!keep_convbn) {
          // Merge Conv + BatchNorm -> Conv
          DLOG(INFO) << " Merge ConvBatchNorm -> Conv: "
                     << net_in.layer(i).name() << " + "
                     << net_in.layer(bn_index).name();
          MergeConvBatchNorm2Conv(net_in, net_out, i, bn_index, processed,
                                  binary);
        } else {
          ; // 
        }
      } else {
        *(net_out.add_layer()) = *cur_layer;
        processed[i] = true;
      }

    } else if (cur_layer->type() == "InnerProduct") {
      int bn_index = GetNextLayer(net_in, i, "BatchNorm", connect);
      if (bn_index != -1) {
        // Merge InnerProduct + BatchNorm -> InnerProduct
        LOG(INFO) << " Merge InnerProductBatchNorm -> InnerProduct: "
                  << net_in.layer(i).name() << " + "
                  << net_in.layer(bn_index).name();
        MergeFCBatchNorm2FC(net_in, net_out, i, bn_index, processed, binary);
      } else {
        *(net_out.add_layer()) = *cur_layer;
        processed[i] = true;
      }

    } else {
      *(net_out.add_layer()) = *cur_layer;
      processed[i] = true;
    }
  }
  return net_out;
}

NetParameter MergePreBatchNormInNet(const NetParameter &net_in, bool binary,
                                    bool keep_convbn) {
  vector<vector<bool>> connect = BuildConnections(net_in);
  vector<bool> processed(net_in.layer_size(), false);
  NetParameter net_out;

  for (int i = 0; i < net_in.layer_size(); i++) {
    if (processed[i]) {
      continue;
    }
    const LayerParameter *cur_layer = &net_in.layer(i);
    // Merge BatchNorm + Convolution
    if (cur_layer->type() == "BatchNorm") {
      bool need_merge = false;
      int conv_index = GetNextLayer(net_in, i, "Convolution", connect);
      int relu_index = GetNextLayer(net_in, i, "ReLU", connect);
      // stop merge for batchnorm + relu + convolution
      // stop merge for batchnorm + convolution_with_pad
      if (conv_index != -1 && (relu_index == -1 || relu_index > conv_index)) {
        auto &conv_param = net_in.layer(conv_index).convolution_param();
        if (conv_param.pad_size()) {
          for (int j = 0; j < conv_param.pad_size(); j++) {
            if (conv_param.pad(j) != 0) {
              need_merge = false;
              break;
            } else {
              need_merge = true;
            }
          }
        } else if (conv_param.has_pad_h() || conv_param.has_pad_w()) {
          need_merge = false;
        } else {
          need_merge = true;
        }
      }

      if (need_merge) {
        if (!keep_convbn) {
          // Merge Conv + BatchNorm -> Conv
          DLOG(INFO) << " Merge BatchNormConv -> Conv: "
                     << net_in.layer(i).name() << " + "
                     << net_in.layer(conv_index).name();
          MergeBatchNormConv2Conv(net_in, net_out, i, conv_index, processed,
                                  binary);
        } else {
          ; //
        }

      } else {
        // Merge BatchNorm -> Scale
        DLOG(INFO) << " Merge BatchNorm -> Scale: " << net_in.layer(i).name();
        MergeBatchNorm2Scale(net_in, net_out, i, processed, binary);
      }

    } else {
      *(net_out.add_layer()) = *cur_layer;
      processed[i] = true;
    }
  }
  return net_out;
}

// This is a function to merge Scale layer into BatchNorm Layer or BatchNorm
// layer into Convolution Layer for prototxt or caffemodel, support caffe1,
// nvcaffe, caffe_parallel's float model.
// Function:
//      1) Convolution + BatchNorm + (Scale) -> Convolution
//      2) (If keep_bn) BatchNorm + Scale -> BatchNorm
//      3) (If keep_convbn) 
int MergeLayers(const NetParameter &net_in, NetParameter &net_out, bool binary,
                bool keep_bn, bool keep_convbn) {
  // 0. InsertSplits
  NetParameter net_split;
  // ignore bottom err in InsertSplits if binary == true
  InsertSplits(net_in, &net_split, binary);

  // 1. Merge BatchNorm + Scale and deal with BVLCBatchNorm/BN
  NetParameter net_1 = MergeBvlcBatchNormInNet(net_split, binary);

  // 2. Merge Convolution + BatchNorm && InnerProduce + BatchNorm
  if (!keep_bn) {
    NetParameter net_2 = MergePostBatchNormInNet(net_1, binary, keep_convbn);
    net_out = MergePreBatchNormInNet(net_2, binary, keep_convbn);
  } else {
    net_out = net_1;
  }
  // 3. RemoveSplits
  net_out = RemoveLayerByType(net_out, "Split");
  return 0;
}

void InitIgnoreFile(YAML::Node &ignore_layers,
                    const string ignore_layers_file_in,
                    const string ignore_layers_in) {
  if (ignore_layers_file_in != "") {
    ignore_layers = YAML::LoadFile(ignore_layers_file_in);
    if (ignore_layers["ignore_layers"].size() == 0) {
      LOG(WARNING) << "Warning: Empty Ingnore Layers List !";
    }
  }
  if (ignore_layers_in != "") {
    size_t start = 0;
    size_t index = ignore_layers_in.find(",", start);
    while (index != std::string::npos) {
      ignore_layers["ignore_layers"].push_back(
          ignore_layers_in.substr(start, index - start));
      start = index + 1;
      index = ignore_layers_in.find(",", start);
    }
    ignore_layers["ignore_layers"].push_back(ignore_layers_in.substr(start));
  }
  for (int i = 0; i < ignore_layers["ignore_layers"].size(); i++) {
    DLOG(INFO) << "Ignore Layers List [" << i
               << "]:" << ignore_layers["ignore_layers"][i];
  }
}

void InitSigmoidedLayers(vector<string> &sigmoided_layers,
                         const string &sigmoided_layers_in) {
  if (sigmoided_layers_in != "") {
    size_t start = 0;
    size_t index = sigmoided_layers_in.find(",", start);
    while (index != std::string::npos) {
      sigmoided_layers.push_back(
          sigmoided_layers_in.substr(start, index - start));
      start = index + 1;
      index = sigmoided_layers_in.find(",", start);
    }
    sigmoided_layers.push_back(sigmoided_layers_in.substr(start));
  }
  for (int i = 0; i < sigmoided_layers.size(); i++) {
    DLOG(INFO) << "Sigmoided Layers List [" << i << "]:" << sigmoided_layers[i];
  }
}

} // namespace caffe
