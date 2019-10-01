#!/bin/bash
cuda_verfile="/usr/local/cuda/version.txt"
cudnn_verfile="/usr/local/cuda/include/cudnn.h"
cuda_ver=
cudnn_ver=

if [ -f $cuda_verfile ] ; then
    cuda_ver=`cat $cuda_verfile | awk '{ print $3 } ' `
    cuda_ver=`echo $cuda_ver | awk -F'.' '{print $1"."$2}'`
fi

if [ -f $cudnn_verfile ] ; then
    cudnn_ver=`cat $cudnn_verfile | grep CUDNN_MAJOR -A 2 | head -3 | awk '{ver=ver$3"."}END{print ver}'`
    cudnn_ver=`echo $cudnn_ver | awk -F'.' '{print $1"."$2"."$3}'`
fi

if [ "$cuda_ver" != "" ]; then
    echo "[CUDA version]"
    echo $cuda_ver
fi

if [ "$cudnn_ver" != "" ]; then
    echo "[CUDNN version]"
    echo $cudnn_ver
fi

rm -rf decent_q_segment

if [ $cuda_ver = 8.0 ] 
then
    echo "setting decent_q_segment for CUDA 8.0, CuDNN 7.05"
    cp decent_q-ubuntu16.04-cuda8.0-cudnn7.0.5 decent_q_segment
elif [ $cuda_ver = 9.0 ] 
then
    echo "setting  decent_q_segment for CUDA 9.0, CuDNN 7.05"
    cp decent_q-ubuntu16.04-cuda9.0-cudnn7.0.5 decent_q_segment
elif [ $cuda_ver = 9.1 ] 
then
    echo "setting  decent_q_segment for CUDA 9.1, CuDNN 7.05"
    cp decent_q-ubuntu16.04-cuda9.1-cudnn7.0.5 decent_q_segment
fi
