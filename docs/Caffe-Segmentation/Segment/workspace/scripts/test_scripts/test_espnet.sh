export PYTHONPATH=../../../caffe-master/python
python test_espnet.py --model ../../model/espnet/deploy.prototxt \
--weights ../../model/espnet/final_models/pretrained.caffemodel \
--colours ../cityscapes19.png --input ../munich_000000_000019_leftImg8bit.png --out_dir ./
