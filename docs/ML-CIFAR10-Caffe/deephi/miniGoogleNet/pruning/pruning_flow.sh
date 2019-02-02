#!/bin/sh

PRUNE_ROOT=$HOME/ML/DNNDK/tools
WORK_DIR=cifar10/deephi/miniGoogleNet/pruning

#take the caffemodel with a (forced) soft link to save HD space
ln -nsf $HOME/ML/cifar10/caffe/models/miniGoogleNet/m3/snapshot_3_miniGoogleNet__iter_40000.caffemodel ${WORK_DIR}/float.caffemodel

# leave commented the next lines, here added only for "documentation" 
#copy the solver and edit it by reducing the amount of iterations and changing the pathnames
#cp $CAFFE_DIR/models/m3/pruning_solver_3_miniGoogleNet.prototxt ./solver.prototxt
#copy the description model and edit it by adding top-1 and top-5 accuracy layers at the bottom and changing the pathnames
#cp $CAFFE_DIR/models/m3/pruning_train_val_3_miniGoogleNet.prototxt ./train_val.prototxt

# analysis: you do it only once
$PRUNE_ROOT/deephi_compress ana -config ${WORK_DIR}/config0.prototxt      2>&1 | tee ${WORK_DIR}/rpt/logfile_ana_miniGoogleNet.txt
 
# compression: zero run
$PRUNE_ROOT/deephi_compress compress -config ${WORK_DIR}/config0.prototxt 2>&1 | tee ${WORK_DIR}/rpt/logfile_compress0_miniGoogleNet.txt
# fine-tuning: zero run
$PRUNE_ROOT/deephi_compress finetune -config ${WORK_DIR}/config0.prototxt 2>&1 | tee ${WORK_DIR}/rpt/logfile_finetune0_miniGoogleNet.txt

# compression: first run
$PRUNE_ROOT/deephi_compress compress -config ${WORK_DIR}/config1.prototxt 2>&1 | tee ${WORK_DIR}/rpt/logfile_compress1_miniGoogleNet.txt
# fine-tuning: first run
$PRUNE_ROOT/deephi_compress finetune -config ${WORK_DIR}/config1.prototxt 2>&1 | tee ${WORK_DIR}/rpt/logfile_finetune1_miniGoogleNet.txt

# compression: second run
$PRUNE_ROOT/deephi_compress compress -config ${WORK_DIR}/config2.prototxt 2>&1 | tee ${WORK_DIR}/rpt/logfile_compress2_miniGoogleNet.txt
## fine-tuning: second run
$PRUNE_ROOT/deephi_compress finetune -config ${WORK_DIR}/config2.prototxt 2>&1 | tee ${WORK_DIR}/rpt/logfile_finetune2_miniGoogleNet.txt

## compression: third run
$PRUNE_ROOT/deephi_compress compress -config ${WORK_DIR}/config3.prototxt 2>&1 | tee ${WORK_DIR}/rpt/logfile_compress3_miniGoogleNet.txt
## fine-tuning: third run
$PRUNE_ROOT/deephi_compress finetune -config ${WORK_DIR}/config3.prototxt 2>&1 | tee ${WORK_DIR}/rpt/logfile_finetune3_miniGoogleNet.txt


## compression: fourth run
$PRUNE_ROOT/deephi_compress compress -config ${WORK_DIR}/config4.prototxt 2>&1 | tee ${WORK_DIR}/rpt/logfile_compress4_miniGoogleNet.txt
## fine-tuning: fourth run
$PRUNE_ROOT/deephi_compress finetune -config ${WORK_DIR}/config4.prototxt 2>&1 | tee ${WORK_DIR}/rpt/logfile_finetune4_miniGoogleNet.txt




## last step: get the final output model
## note that it does not work if you used the "final.prototxt" as wrongly described by transform help
#
$PRUNE_ROOT/deephi_compress transform -model ${WORK_DIR}/train_val.prototxt -weights ${WORK_DIR}/regular_rate_0.4/snapshots/_iter_40000.caffemodel 2>&1 | tee ${WORK_DIR}/rpt/logfile_transform_miniGoogleNet.txt
mv transformed.caffemodel ${WORK_DIR}

# get flops and the number of parameters of a model
$PRUNE_ROOT/deephi_compress stat -model ${WORK_DIR}/train_val.prototxt 2>&1 | tee ${WORK_DIR}/rpt/logfile_stat_miniGoogleNet.txt



