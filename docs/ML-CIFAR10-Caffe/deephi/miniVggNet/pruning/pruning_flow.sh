#!/bin/bash

ML_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && cd ../../.. && pwd )"
export ML_DIR

PRUNE_ROOT=/usr/local/bin
WORK_DIR=$ML_DIR/deephi/miniVggNet/pruning

[ -f /usr/local/bin/deephi_compress ] || PRUNE_ROOT=$HOME/ML/DNNDK/tools

#take the caffemodel with a (forced) soft link to save HD space
ln -nsf $ML_DIR/caffe/models/miniVggNet/m3/snapshot_3_miniVggNet__iter_40000.caffemodel ${WORK_DIR}/float.caffemodel

# leave commented the next lines, here added only for "documentation" 

#copy the solver and edit it by reducing the amount of iterations and changing the pathnames
##cp $ML_DIR/caffe/models/miniVggNet/m3/solver_3_miniVggNet.prototxt ./solver.prototxt

#copy the description model and edit it by adding top-1 and top-5 accuracy layers at the bottom and changing the pathnames
##cp $ML_DIR/caffe/models/miniVggNet/m3/train_val_3_miniVggNet.prototxt ./train_val.prototxt

# analysis: you do it only once
${PRUNE_ROOT}/deephi_compress ana -config ${WORK_DIR}/config0.prototxt      2>&1 | tee ${WORK_DIR}/rpt/logfile_ana_miniVggNet.txt

# compression: zero run
${PRUNE_ROOT}/deephi_compress compress -config ${WORK_DIR}/config0.prototxt 2>&1 | tee ${WORK_DIR}/rpt/logfile_compress0_miniVggNet.txt
# fine-tuning: zero run
${PRUNE_ROOT}/deephi_compress finetune -config ${WORK_DIR}/config0.prototxt 2>&1 | tee ${WORK_DIR}/rpt/logfile_finetune0_miniVggNet.txt


# compression: first run
${PRUNE_ROOT}/deephi_compress compress -config ${WORK_DIR}/config1.prototxt 2>&1 | tee ${WORK_DIR}/rpt/logfile_compress1_miniVggNet.txt
# fine-tuning: first run
${PRUNE_ROOT}/deephi_compress finetune -config ${WORK_DIR}/config1.prototxt 2>&1 | tee ${WORK_DIR}/rpt/logfile_finetune1_miniVggNet.txt

# compression: second run
${PRUNE_ROOT}/deephi_compress compress -config ${WORK_DIR}/config2.prototxt 2>&1 | tee ${WORK_DIR}/rpt/logfile_compress2_miniVggNet.txt
## fine-tuning: second run
${PRUNE_ROOT}/deephi_compress finetune -config ${WORK_DIR}/config2.prototxt 2>&1 | tee ${WORK_DIR}/rpt/logfile_finetune2_miniVggNet.txt

## compression: third run
${PRUNE_ROOT}/deephi_compress compress -config ${WORK_DIR}/config3.prototxt 2>&1 | tee ${WORK_DIR}/rpt/logfile_compress3_miniVggNet.txt
## fine-tuning: third run
${PRUNE_ROOT}/deephi_compress finetune -config ${WORK_DIR}/config3.prototxt 2>&1 | tee ${WORK_DIR}/rpt/logfile_finetune3_miniVggNet.txt

## compression: fourth run
${PRUNE_ROOT}/deephi_compress compress -config ${WORK_DIR}/config4.prototxt 2>&1 | tee ${WORK_DIR}/rpt/logfile_compress4_miniVggNet.txt
## fine-tuning: fourth run
${PRUNE_ROOT}/deephi_compress finetune -config ${WORK_DIR}/config4.prototxt 2>&1 | tee ${WORK_DIR}/rpt/logfile_finetune4_miniVggNet.txt

## compression: fifth run
${PRUNE_ROOT}/deephi_compress compress -config ${WORK_DIR}/config5.prototxt 2>&1 | tee ${WORK_DIR}/rpt/logfile_compress5_miniVggNet.txt
## fine-tuning: fifth run
${PRUNE_ROOT}/deephi_compress finetune -config ${WORK_DIR}/config5.prototxt 2>&1 | tee ${WORK_DIR}/rpt/logfile_finetune5_miniVggNet.txt

## compression: 6-th run
${PRUNE_ROOT}/deephi_compress compress -config ${WORK_DIR}/config6.prototxt 2>&1 | tee ${WORK_DIR}/rpt/logfile_compress6_miniVggNet.txt
## fine-tuning: 6-th run
${PRUNE_ROOT}/deephi_compress finetune -config ${WORK_DIR}/config6.prototxt 2>&1 | tee ${WORK_DIR}/rpt/logfile_finetune6_miniVggNet.txt

## compression: 7-th run
${PRUNE_ROOT}/deephi_compress compress -config ${WORK_DIR}/config7.prototxt 2>&1 | tee ${WORK_DIR}/rpt/logfile_compress7_miniVggNet.txt
## fine-tuning: 7-th run
${PRUNE_ROOT}/deephi_compress finetune -config ${WORK_DIR}/config7.prototxt 2>&1 | tee ${WORK_DIR}/rpt/logfile_finetune7_miniVggNet.txt

## etc


## last step: get the final output model
## note that it does not work if you used the "final.prototxt" as wrongly described by transform help
#
${PRUNE_ROOT}/deephi_compress transform -model ${WORK_DIR}/train_val.prototxt -weights ${WORK_DIR}/regular_rate_0.7/snapshots/_iter_20000.caffemodel 2>&1 | tee ${WORK_DIR}/rpt/logfile_transform_miniVggNet.txt

# get flops and the number of parameters of a model
${PRUNE_ROOT}/deephi_compress stat -model ${WORK_DIR}/train_val.prototxt 2>&1 | tee ${WORK_DIR}/rpt/logfile_stat_miniVggNet.txt

for file in $(find $ML_DIR -name transformed.caffemodel); do
    mv ${file} ${WORK_DIR}
done

