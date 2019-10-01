#ifndef CAFFE_UTIL_CONVERT_PROTO_HPP_
#define CAFFE_UTIL_CONVERT_PROTO_HPP_

#include "caffe/blob.hpp"
#include "caffe/caffe.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/io.hpp"
#include "opencv2/opencv.hpp"
#include "yaml-cpp/yaml.h"
#include <unordered_set>

namespace caffe {

NetParameter RemoveLayerByType(const NetParameter &model_in, const string type);

NetParameter RemoveLayerByPhase(const NetParameter &model_in,
                                caffe::Phase phase);

NetParameter RemoveLayerByName(const NetParameter &model_in, const string name);

NetParameter RemoveBlobs(const NetParameter &model_in);

bool CheckImageDataLayer(const LayerParameter *data_layer);

bool CheckModel(const NetParameter &net_param);

bool CheckModelAndSave(const NetParameter &net_param, const string filename);

template <typename Dtype>
void DataStat(const int n, const Dtype *data, const char* title);

template <typename Dtype>
void BindWeightWithProfile(Blob<float>& new_weight, const Dtype *weight,
                const Dtype *scale, const Dtype *var, float eps);

template <typename Dtype>
void BindWeight(const int n, const int kernel_dim, const Dtype *weight,
                const Dtype *scale, const Dtype *var, Dtype *bind_weight, float eps);

template <typename Dtype>
void BindBias(const int n, const Dtype *mean, const Dtype *scale,
              const Dtype *var, const Dtype *bias_conv, const Dtype *bias_bn,
              Dtype *bind_bias, float eps);

template <typename Dtype>
void ScaleInvVar(const int n, const Dtype *scale, const Dtype *var,
                 Dtype *scale_inv_var, float eps);

vector<vector<bool>> BuildConnections(const NetParameter &net);

void ShowConnections(NetParameter net, vector<vector<bool>> connect);

const string InferNetType(const NetParameter &net_param);

const vector<int> GetInputShape(NetParameter net_param, const string &input_blob);

TransformationParameter GetTransformParam(const NetParameter &net);

int GetNextLayer(const NetParameter &net, const int index, const string type,
                 const vector<vector<bool>> &connect);

int GetPreviousLayer(const NetParameter &net, const int index,
                     const string type, const vector<vector<bool>> &connect);

bool IsInplace(const LayerParameter *layer);

int GetLastInputLayer(const NetParameter &net);

// Merge Model
void MergeConvBatchNorm2Conv(const NetParameter &net_in, NetParameter &net_out,
                             const int conv_index, const int bn_index,
                             vector<bool> &processed, bool binary = false);

void MergeBatchNormConv2Conv(const NetParameter &net_in, NetParameter &net_out,
                             const int bn_index, const int conv_index,
                             vector<bool> &processed, bool binary = false);

void MergeFCBatchNorm2FC(const NetParameter &net_in, NetParameter &net_out,
                             const int fc_index, const int bn_index,
                             vector<bool> &processed, bool binary = false);

void MergeBvlcBatchNormScale2BatchNorm(const NetParameter &net_in,
                                       NetParameter &net_out,
                                       const int bn_index, const int sc_index,
                                       vector<bool> &processed,
                                       bool binary = false);

void MergeBatchNorm2Scale(const NetParameter &net_in, NetParameter &net_out,
                          const int bn_index, vector<bool> &processed,
                          bool binary = false);

void ConvertBvlcBatchNorm(const NetParameter &net_in, NetParameter &net_out,
                          const int bn_index, vector<bool> &processed,
                          bool binary = false);

void ConvertBN2BatchNorm(const NetParameter &net_in, NetParameter &net_out,
                         const int bn_index, vector<bool> &processed,
                         bool binary = false);

// Convert train2deploy
void ConvertSoftLoss2Soft(const NetParameter &net, const int soft_index);

NetParameter ConvertTrain2Deploy(const NetParameter &train_net);

int MergeLayers(const NetParameter &net_in, NetParameter &net_out, bool binary,
                bool keep_bn, bool keep_convbn);

void InitIgnoreFile(YAML::Node &ignore_layers,
                    const string ignore_layers_file_in,
                    const string ignore_layers_in);

void InitSigmoidedLayers(vector<string> &sigmoided_layers,
                    const string &ignore_layers_in);

} // namespace caffe
#endif // CAFFE_UTIL_CONVERT_PROTO_HPP_
