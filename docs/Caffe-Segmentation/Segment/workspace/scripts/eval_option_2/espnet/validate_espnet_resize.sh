export GPUID=1
export PYTHONPATH=../../../../caffe-master/python
if [[ -z "${CITYSCAPES_DATASET}" ]]; then
  export CITYSCAPES_DATASET=/data2/datasets/Cityscapes
fi
export CITYSCAPES_RESULTS=`pwd`/results
rm -rf results_espnet/frankfurt/*
rm -rf results_espnet/lindau/*
rm -rf results_espnet/munster/*
python validate_espnet_resize.py --model deploy_espnet_resize.prototxt --weights ../../../model/espnet/final_models/pretrained.caffemodel --colours ../../cityscapes19.png --input_directory ${CITYSCAPES_DATASET}/leftImg8bit/val/frankfurt --out_dir ./results/frankfurt/ --gpu $GPUID
python validate_espnet_resize.py --model deploy_espnet_resize.prototxt --weights ../../../model/espnet/final_models/pretrained.caffemodel --colours ../../cityscapes19.png --input_directory ${CITYSCAPES_DATASET}/leftImg8bit/val/lindau --out_dir ./results/lindau/ --gpu $GPUID
python validate_espnet_resize.py --model deploy_espnet_resize.prototxt --weights ../../../model/espnet/final_models/pretrained.caffemodel --colours ../../cityscapes19.png --input_directory ${CITYSCAPES_DATASET}/leftImg8bit/val/munster --out_dir ./results/munster/ --gpu $GPUID
csEvalPixelLevelSemanticLabeling 2>&1 | tee results.txt