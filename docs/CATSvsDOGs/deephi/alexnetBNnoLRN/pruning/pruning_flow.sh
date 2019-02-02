#!/bin/sh

PRUNE_ROOT=$HOME/ML/DNNDK/tools
WORK_DIR=$HOME/ML/cats-vs-dogs/deephi/alexnetBNnoLRN/pruning

#take the caffemodel with a soft link to save HD space
ln -nsf $HOME/ML/cats-vs-dogs/caffe/models/alexnetBNnoLRN/m2/snapshot_2_alexnetBNnoLRN__iter_12000.caffemodel  ${WORK_DIR}/float.caffemodel
#ln -nsf $HOME/ML/cats-vs-dogs/caffe/models/alexnetBNnoLRN/m2/rpt/aws_float.caffemodel  ${WORK_DIR}/float.caffemodel

# analysis: you do it only once
$PRUNE_ROOT/deephi_compress ana -config ${WORK_DIR}/config0.prototxt      2>&1 | tee ${WORK_DIR}/rpt/logfile_ana_alexnetBNnoLRN.txt

# compression: zero run
$PRUNE_ROOT/deephi_compress compress -config ${WORK_DIR}/config0.prototxt 2>&1 | tee ${WORK_DIR}/rpt/logfile_compress0_alexnetBNnoLRN.txt
# fine-tuning: zero run
$PRUNE_ROOT/deephi_compress finetune -config ${WORK_DIR}/config0.prototxt 2>&1 | tee ${WORK_DIR}/rpt/logfile_finetune0_alexnetBNnoLRN.txt

# compression: first run
$PRUNE_ROOT/deephi_compress compress -config ${WORK_DIR}/config1.prototxt 2>&1 | tee ${WORK_DIR}/rpt/logfile_compress1_alexnetBNnoLRN.txt
# fine-tuning: first run
$PRUNE_ROOT/deephi_compress finetune -config ${WORK_DIR}/config1.prototxt 2>&1 | tee ${WORK_DIR}/rpt/logfile_finetune1_alexnetBNnoLRN.txt

# compression: second run
$PRUNE_ROOT/deephi_compress compress -config ${WORK_DIR}/config2.prototxt 2>&1 | tee ${WORK_DIR}/rpt/logfile_compress2_alexnetBNnoLRN.txt
## fine-tuning: second run
$PRUNE_ROOT/deephi_compress finetune -config ${WORK_DIR}/config2.prototxt 2>&1 | tee ${WORK_DIR}/rpt/logfile_finetune2_alexnetBNnoLRN.txt

## compression: third run
$PRUNE_ROOT/deephi_compress compress -config ${WORK_DIR}/config3.prototxt 2>&1 | tee ${WORK_DIR}/rpt/logfile_compress3_alexnetBNnoLRN.txt
## fine-tuning: third run
$PRUNE_ROOT/deephi_compress finetune -config ${WORK_DIR}/config3.prototxt 2>&1 | tee ${WORK_DIR}/rpt/logfile_finetune3_alexnetBNnoLRN.txt

## compression: fourth run
$PRUNE_ROOT/deephi_compress compress -config ${WORK_DIR}/config4.prototxt 2>&1 | tee ${WORK_DIR}/rpt/logfile_compress4_alexnetBNnoLRN.txt
## fine-tuning: fourth run
$PRUNE_ROOT/deephi_compress finetune -config ${WORK_DIR}/config4.prototxt 2>&1 | tee ${WORK_DIR}/rpt/logfile_finetune4_alexnetBNnoLRN.txt

## compression: fift run
$PRUNE_ROOT/deephi_compress compress -config ${WORK_DIR}/config5.prototxt 2>&1 | tee ${WORK_DIR}/rpt/logfile_compress5_alexnetBNnoLRN.txt
## fine-tuning: fift run
$PRUNE_ROOT/deephi_compress finetune -config ${WORK_DIR}/config5.prototxt 2>&1 | tee ${WORK_DIR}/rpt/logfile_finetune5_alexnetBNnoLRN.txt

## compression: 6-th run
$PRUNE_ROOT/deephi_compress compress -config ${WORK_DIR}/config6.prototxt 2>&1 | tee ${WORK_DIR}/rpt/logfile_compress6_alexnetBNnoLRN.txt
## fine-tuning: 6-th run
$PRUNE_ROOT/deephi_compress finetune -config ${WORK_DIR}/config6.prototxt 2>&1 | tee ${WORK_DIR}/rpt/logfile_finetune6_alexnetBNnoLRN.txt

## compression: 7-th run
$PRUNE_ROOT/deephi_compress compress -config ${WORK_DIR}/config7.prototxt 2>&1 | tee ${WORK_DIR}/rpt/logfile_compress7_alexnetBNnoLRN.txt
## fine-tuning: 7-th run
$PRUNE_ROOT/deephi_compress finetune -config ${WORK_DIR}/config7.prototxt 2>&1 | tee ${WORK_DIR}/rpt/logfile_finetune7_alexnetBNnoLRN.txt

## last step: get the final output model
## note that it does not work if you used the "final.prototxt" as wrongly described by transform help
$PRUNE_ROOT/deephi_compress transform -model ${WORK_DIR}/train_val.prototxt -weights ${WORK_DIR}/regular_rate_0.7/sparse.caffemodel 2>&1 | tee ${WORK_DIR}/rpt/logfile_transform_alexnetBNnoLRN.txt

# get flops and the number of parameters of a model
$PRUNE_ROOT/deephi_compress stat -model ${WORK_DIR}/train_val.prototxt 2>&1 | tee ${WORK_DIR}/rpt/logfile_stat_alexnetBNnoLRN.txt

mv transformed.caffemodel ${WORK_DIR}/
