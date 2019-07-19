#!/bin/bash


conda activate decent_q3

echo "#####################################"
echo "TRAIN & SAVE"
echo "#####################################"
python train_save.py


echo "#####################################"
echo "CONVERT KERAS TO TF"
echo "#####################################"
# method 1
python keras_2_tf.py --keras_hdf5 ./method1/keras_chkpt.h5 \
                     --tfckpt=./method1/tf_chkpt.ckpt  \
                     --tf_graph=./method1/tf_infer_graph.pb

# method 2
python keras_2_tf.py --keras_hdf5 ./method2/k_complete_model.h5 \
                     --tfckpt=./method2/tf_chkpt.ckpt  \
                     --tf_graph=./method2/tf_infer_graph.pb

# method 3
python keras_2_tf.py --keras_json=./method3/k_model_architecture.json \
                     --keras_hdf5=./method3/k_model_weights.h5 \
                     --tfckpt=./method3/tf_chkpt.ckpt  \
                     --tf_graph=./method3/tf_infer_graph.pb


echo "#####################################"
echo "FREEZE GRAPH"
echo "#####################################"
# method 1
freeze_graph --input_graph=./method1/tf_infer_graph.pb \
             --input_checkpoint=./method1/tf_chkpt.ckpt \
             --input_binary=true \
             --output_graph=./method1/frozen_graph.pb \
             --output_node_names=dense_2/Softmax

# method 2
freeze_graph --input_graph=./method2/tf_infer_graph.pb \
             --input_checkpoint=./method2/tf_chkpt.ckpt \
             --input_binary=true \
             --output_graph=./method2/frozen_graph.pb \
             --output_node_names=dense_2/Softmax

# method 3
freeze_graph --input_graph=./method3/tf_infer_graph.pb \
             --input_checkpoint=./method3/tf_chkpt.ckpt \
             --input_binary=true \
             --output_graph=./method3/frozen_graph.pb \
             --output_node_names=dense_2/Softmax


echo "#####################################"
echo "QUANTIZE"
echo "#####################################"
# method 1
decent_q quantize \
        --input_frozen_graph=./method1/frozen_graph.pb \
        --input_nodes=input_1 \
        --input_shapes=?,32,32,3 \
        --output_nodes=dense_2/Softmax \
        --method=1 \
        --input_fn=random \
        --output_dir=method1 \
        --gpu=0

# method 2
decent_q quantize \
        --input_frozen_graph=./method2/frozen_graph.pb \
        --input_nodes=input_1 \
        --input_shapes=?,32,32,3 \
        --output_nodes=dense_2/Softmax \
        --method=1 \
        --input_fn=random \
        --output_dir=method2 \
        --gpu=0

# method 3
decent_q quantize \
        --input_frozen_graph=./method3/frozen_graph.pb \
        --input_nodes=input_1 \
        --input_shapes=?,32,32,3 \
        --output_nodes=dense_2/Softmax \
        --method=1 \
        --input_fn=random \
        --output_dir=method3 \
        --gpu=0

echo "#####################################"
echo "COMPILE"
echo "#####################################"
# method 1
dnnc   --parser=tensorflow \
       --frozen_pb=./method1/deploy_model.pb \
       --dpu=4096FA \
       --cpu_arch=arm64 \
       --output_dir=method1 \
       --save_kernel \
       --mode normal \
       --net_name=cifar10_net

# method 2
dnnc   --parser=tensorflow \
       --frozen_pb=./method2/deploy_model.pb \
       --dpu=4096FA \
       --cpu_arch=arm64 \
       --output_dir=method2 \
       --save_kernel \
       --mode normal \
       --net_name=cifar10_net

# method 3
dnnc   --parser=tensorflow \
       --frozen_pb=./method3/deploy_model.pb \
       --dpu=4096FA \
       --cpu_arch=arm64 \
       --output_dir=method3 \
       --save_kernel \
       --mode normal \
       --net_name=cifar10_net








