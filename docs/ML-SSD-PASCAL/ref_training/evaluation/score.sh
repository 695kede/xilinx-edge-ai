./build/tools/caffe train \
--solver="evaluation/solver_score.prototxt" \
--weights="models/VGGNet/VOC0712/SSD_300x300/VGG_VOC0712_SSD_300x300_iter_120000.caffemodel" \
--log_dir="evaluation/score_results" \
--gpu 0  | tee evaluation/score_results/VGG_VOC0712_SSD_300x300.log \


