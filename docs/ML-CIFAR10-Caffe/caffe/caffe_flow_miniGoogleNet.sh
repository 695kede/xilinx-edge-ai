#!/bin/sh

#working dir (i.e. "/home/danieleb/ML/cifar10")
WORK_DIR=$HOME/ML/cifar10/caffe

# adjust this $CAFFE_ROOT to point your Caffe install dir. It must match the CAFFE_ROOT of "cifar10_config.py"
CAFFE_ROOT=$HOME/caffe_tools/BVLC1v0-Caffe
CAFFE_TOOLS_DIR=$CAFFE_ROOT/distribute

MOD_NUM=3   # model number
NUMIT=40000 # number of iterations
NET=miniGoogleNet

: '
# ################################################################################################################
# SCRIPTS 1 2 3 (DATABASE AND MEAN VALUES)
echo "DATABASE: training and validation in LMDB, test in JPG and MEAN values"

# load the database from keras and write it as JPEG images
python $WORK_DIR/code/1_write_cifar10_images.py

#create LMDB databases -training (50K), validation (9K), test (1K) images - and compute mean values
python $WORK_DIR/code/2a_create_lmdb.py  
python $WORK_DIR/code/2b_compute_mean.py 

#check goodness of LMDB databases (just for debug: you can skip it)
python $WORK_DIR/code/3_read_lmdb.py
'

 ################################################################################################################
# SCRIPT 4  (SOLVER AND TRAINING AND LEARNING CURVE)
echo "TRAINING. Remember that: <Epoch_index = floor((iteration_index * batch_size) / (# data_samples))>"

#$CAFFE_TOOLS_DIR/bin/caffe.bin train --solver $WORK_DIR/models/$NET/m$MOD_NUM/solver_$MOD_NUM\_$NET.prototxt \
#			       2>&1 | tee $WORK_/models/$NET/m$MOD_NUM/logfile_$MOD_NUM\_$NET.log

cd $HOME/ML  #you must launch this script from your $HOME (i.e. "/home/danieleb")

python $WORK_DIR/code/4_training.py -s $WORK_DIR/models/$NET/m$MOD_NUM/solver_$MOD_NUM\_$NET.prototxt -l $WORK_DIR/models/$NET/m$MOD_NUM/logfile_$MOD_NUM\_$NET.log


: '
# # example of trainining the CNN from a certain snapshot
#echo "RETRAINING from previous snapshot"
# #$CAFFE_TOOLS_DIR/bin/caffe.bin train --solver $WORK_DIR/models/$NET/m$MOD_NUM/solver_$MOD_NUM\_$NET.prototxt \
#      --snapshot $WORK_DIR/models/$NET/m3/snapshot_3\$NET__iter_20000.solverstate \
#      2>&1 | tee $WORK_DIR/models/$NET/m$MOD_NUM/retrain_logfile_$MOD_NUM\_$NET.log
#cp -f $WORK_DIR/models/$NET/m$MOD_NUM/logfile_$MOD_NUM\_$NET.log $WORK_DIR/models/$NET/m$MOD_NUM/orig_logfile_$MOD_NUM\_$NET.log
#cp -f $WORK_DIR/models/$NET/m$MOD_NUM/retrain_logfile_$MOD_NUM\_$NET.log $WORK_DIR/models/$NET/m$MOD_NUM/logfile_$MOD_NUM\_$NET.log
'

# print image of CNN architecture
echo "PRINT CNN BLOCK DIAGRAM"
python $CAFFE_TOOLS_DIR/python/draw_net.py $WORK_DIR/models/$NET/m$MOD_NUM/train_val_$MOD_NUM\_$NET.prototxt $WORK_DIR/models/$NET/m$MOD_NUM/bd_$MOD_NUM\_$NET.png

# ################################################################################################################
# SCRIPT 5: plot the learning curve
echo "PLOT LEARNING CURVERS"
python $WORK_DIR/code/5_plot_learning_curve.py $WORK_DIR/models/$NET/m$MOD_NUM/logfile_$MOD_NUM\_$NET.log $WORK_DIR/models/$NET/m$MOD_NUM/plt_train_val_$MOD_NUM\_$NET.png

# 0 Test Accuracy vs Iters
# 1 Test Accuracy vs Seconds
# 2 Test Loss     vs Iters
# 3 Test Loss     vs Seconds
# 4 Train lr      vs. Iters
# 5 Train lr      vs. Seconds 
# 6 Train Loss     vs Iters
# 7 Train Loss     vs Seconds
python $WORK_DIR/code/plot_training_log.py 6 $WORK_DIR/models/$NET/m$MOD_NUM/plt_trainLoss_$MOD_NUM\_$NET.png     $WORK_DIR/models/$NET/m$MOD_NUM/logfile_$MOD_NUM\_$NET.log
python $WORK_DIR/code/plot_training_log.py 2 $WORK_DIR/models/$NET/m$MOD_NUM/plt_testLoss_$MOD_NUM\_$NET.png      $WORK_DIR/models/$NET/m$MOD_NUM/logfile_$MOD_NUM\_$NET.log
python $WORK_DIR/code/plot_training_log.py 0 $WORK_DIR/models/$NET/m$MOD_NUM/plt_testAccuracy_$MOD_NUM\_$NET.png  $WORK_DIR/models/$NET/m$MOD_NUM/logfile_$MOD_NUM\_$NET.log


# ################################################################################################################
# SCRIPT 6 (PREDICTION)
echo "COMPUTE PREDICTIONS"
python $WORK_DIR/code/6_make_predictions.py -d cifar10/caffe/models/$NET/m$MOD_NUM/deploy_$MOD_NUM\_$NET.prototxt -w cifar10/caffe/models/$NET/m$MOD_NUM/snapshot_$MOD_NUM\_$NET\__iter_$NUMIT.caffemodel 2>&1 | tee cifar10/caffe/models/$NET/m$MOD_NUM/predictions_$MOD_NUM\_$NET.txt


