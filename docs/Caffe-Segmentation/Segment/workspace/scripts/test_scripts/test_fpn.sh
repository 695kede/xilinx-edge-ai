export PYTHONPATH=../../../caffe-master/python
python test_fpn.py --model ../../model/FPN/deploy.prototxt \
--weights ../../model/FPN/final_models/pretrained.caffemodel \
--colours ../cityscapes19.png --input ../munich_000000_000019_leftImg8bit.png --out_dir ./
