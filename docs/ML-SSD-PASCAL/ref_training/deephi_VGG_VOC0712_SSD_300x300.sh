cd /home/danieleb/caffe_tools/Caffe-SSD-Ristretto
./build/tools/caffe train \
--solver="$CAFFE_ROOT/models/VGGNet/VOC0712/SSD_300x300/deephi_solver.prototxt" \
--gpu 0 2>&1 | tee jobs/VGGNet/VOC0712/SSD_300x300/deephi_VGG_VOC0712_SSD_300x300.log

#--snapshot="$CAFFE_ROOT/models/VGGNet/VOC0712/SSD_300x300/VGG_VOC0712_SSD_300x300_iter_23.solverstate" \
