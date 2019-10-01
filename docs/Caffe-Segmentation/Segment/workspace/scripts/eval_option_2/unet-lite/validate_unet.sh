export GPUID=0
export PYTHONPATH=../../../../caffe-master/python
if [[ -z "${CITYSCAPES_DATASET}" ]]; then
  export CITYSCAPES_DATASET=/data2/datasets/Cityscapes
fi
export CITYSCAPES_RESULTS=`pwd`/results
rm -rf results/frankfurt/*
rm -rf results/lindau/*
rm -rf results/munster/*
python validate_unet.py --model deploy_unet.prototxt --weights ../../../model/unet-lite/final_models/pretrained.caffemodel --colours ../../cityscapes19.png --input_directory ${CITYSCAPES_DATASET}/leftImg8bit/val/frankfurt --out_dir ./results/frankfurt/ --gpu $GPUID
python validate_unet.py --model deploy_unet.prototxt --weights ../../../model/unet-lite/final_models/pretrained.caffemodel --colours ../../cityscapes19.png --input_directory ${CITYSCAPES_DATASET}/leftImg8bit/val/lindau --out_dir ./results/lindau/ --gpu $GPUID
python validate_unet.py --model deploy_unet.prototxt --weights ../../../model/unet-lite/final_models/pretrained.caffemodel --colours ../../cityscapes19.png --input_directory ${CITYSCAPES_DATASET}/leftImg8bit/val/munster --out_dir ./results/munster/ --gpu $GPUID
csEvalPixelLevelSemanticLabeling 2>&1 | tee results.txt