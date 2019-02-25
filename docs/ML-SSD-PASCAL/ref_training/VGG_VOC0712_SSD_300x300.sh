cd $CAFFE_ROOT
./build/tools/caffe train \
--solver="jobs/VGGNet/VOC0712/SSD_300x300/solver.prototxt" \
--weights="models/VGGNet/VGG_ILSVRC_16_layers_fc_reduced.caffemodel" \
--log_dir="/data2/caffe/logs" \
--gpu 0  | tee jobs/VGGNet/VOC0712/SSD_300x300/VGG_VOC0712_SSD_300x300.log \

