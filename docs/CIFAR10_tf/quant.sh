#!/bin/bash


# activate DECENT_Q Python3.6 virtual environment
conda activate decent_q3

# generate calibraion images and list file
python generate_images.py

# remove existing files
rm -rf ./quantize_results


# run quantization
echo "#####################################"
echo "QUANTIZE"
echo "#####################################"
decent_q quantize \
  --input_frozen_graph ./freeze/frozen_graph.pb \
  --input_nodes images_in \
  --input_shapes ?,32,32,3 \
  --output_nodes dense_1/BiasAdd \
  --method 1 \
  --input_fn default \
  --calib_iter 100 \
  --batch_size 50 \
  --image_dir ./calib_dir \
  --image_list ./calib_dir/calib_list.txt \
  --scales 0.00392,0.00392,0.00392 \
  --gpu 0

echo "#####################################"
echo "QUANTIZATION COMPLETED"
echo "#####################################"

