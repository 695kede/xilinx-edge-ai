export PYTHONPATH=../../../caffe-master/python
python test_enet.py --model ../../model/enet/deploy.prototxt \
--weights ../../model/enet/final_models/pretrained.caffemodel \
--colours ../cityscapes19.png \
--input ../munich_000000_000019_leftImg8bit.png --out_dir ./ --gpu 0
