
<div style="page-break-after: always;"></div>
<table style="width:100%">
  <tr>
    <th width="100%" colspan="6"><img src="https://www.xilinx.com/content/dam/xilinx/imgs/press/media-kits/corporate/xilinx-logo.png" width="30%"/><h1>Freezing a Keras Model to use with DNNDK</h2>
</th>
  </tr>
  <tr>
  </tr>
</table>
</div>

# Introduction

You must supply a frozen graph in the binary protobuf format (usually has a `.pb` file extension) when using TensorFlow and the deep neural network development kit (DNNDK). Generating a `pb` file in TensorFlow is simple: Save an inference graph and a TensorFlow checkpoint, then run the `freeze_graph` script that comes with TensorFlow.

However, generating a `.pb` file is not so straightforward in Keras because the native 'save' format for Keras is HDF5 or a mix of JSON and HDF5. These formats have to be translated into the binary protobuf format.

You can save a Keras model in three ways:

1. Method 1: By using HDF5 checkpoints during training.
2. Method 2: By saving the complete model in the HDF5 format.
3. Method 3: By saving the network architecture in the JSON format and the weights, biases, and other parameters in the HDF5 format.

>**:warning: WARNING** TensorFlow checkpoints and Keras checkpoints are **_not_** the same!

Methods 1 and 2 save a complete training model, including the network architecture, its current weights and biases, and training operations, such as the loss and optimizer functions, and their current states. This allows you to resume the training from its current point, if required.

Method 3 creates a JSON file which describes the network architecture. The weights are saved into a separate HDF5 file. The training information is not saved, and so this format cannot be used for resuming training. However, it produces smaller files and is, thus, favored when going into deployment.

This repository contains examples for all three methods:

+ `train_save.py` trains a simple network to classify the CIFAR-10 dataset and saves the trained model using the three methods outlined in this application note.
   >**:pushpin: NOTE** This will only run for a few epochs and the accuracy will be quite low.

+ `keras_2_tf.py` creates TensorFlow checkpoints and inference graphs from the saved Keras models.
+ `run_all.sh` runs the complete flow.

# Method 1: Keras Checkpoint to DNNDK

If you are starting at a Keras checkpoint, begin by accessing the underlying TensorFlow backend using the `backend` API. Then, set the learning phase to '0' to indicate to layers, such as dropout or batch normalization, that you are no longer training, and finally, load the checkpoint into a new model using `load_model`, as shown in the following code segment:

```python
# set learning phase for no training
backend.set_learning_phase(0)

# load weights & architecture into new model
loaded_model = load_model('keras_chkpt.h5')
```

Now, you may write out the TensorFlow compatible checkpoint and inference graph which will later be used with the `freeze_graph.py` script to create the frozen model:

```python
# make list of output node names
output_names=[out.op.name for out in loaded_model.outputs]

# set up tensorflow saver object
saver = tf.train.Saver()

# fetch the tensorflow session using the Keras backend
tf_session = backend.get_session()

# get the tensorflow session graph
input_graph_def = tf_session.graph.as_graph_def()

# write out tensorflow checkpoint & inference graph for use with freeze_graph script
save_path = saver.save(tf_session, 'tf_chkpt.ckpt')
tf.train.write_graph(input_graph_def, 'checkpoint_dir', 'tf_infer_graph.pb', as_text=False)
```

# Method 2: Keras Model to DNNDK

If you save the Keras model as a complete model using `save_model`, as shown below, you will have an HDF5 file that is identical to the one produced when the Keras checkpoint was created. In this case, follow the method described in Method 1.

```python
# save weights, model architecture & optimizer to an HDF5 format file
model.save('k_complete_model.h5')
```
# Method 3: JSON and HDF5 to DNNDK

If you save the Keras model as a JSON file for the architecture and an HDF5 file for the weights and biases using `save_weight`, as shown below, you must first recreate the model.

```python
# save just the weights (no architecture) to an HDF5 format file
model.save_weights('k_model_weights.h5')

# save just the architecture (no weights) to a JSON file
with open('k_model_architecture.json', 'w') as f:
    f.write(model.to_json())
```
To recreate the model, first read the JSON file and then load the model with the weights stored in the HDF5 file, as shown below:

```python
# set learning phase for no training
backend.set_learning_phase(0)

# load json and create model
json_file = open('k_model_architecture.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

# load weights into new model
loaded_model.load_weights('k_model_weights.h5')
```
Now, you may write out the TensorFlow compatible checkpoint and inference graph which will be used later with the `freeze_graph.py` script to create the frozen model:

```python
# make list of output node names
output_names=[out.op.name for out in loaded_model.outputs]

# set up tensorflow saver object
saver = tf.train.Saver()

# fetch the tensorflow session using the Keras backend
tf_session = backend.get_session()

# get the tensorflow session graph
input_graph_def = tf_session.graph.as_graph_def()

# write out tensorflow checkpoint & inference graph for use with freeze_graph script
save_path = saver.save(tf_session, 'tf_chkpt.ckpt')
tf.train.write_graph(input_graph_def, 'checkpoint_dir', 'tf_infer_graph.pb', as_text=False)
```

# Running the Tutorial using Scripts

1. Clone the repository, open a terminal, and run the `cd` command to go to the repository folder that you just created.
2. Run the complete flow using the `source ./run_all.sh` script.

>**:pushpin: NOTE**
>The `run_all.sh` script contain references to a python virtual environment that is handled by Anaconda, as shown below:
>
>```
>conda activate decent_q3
>```
>
 >The names of the virtual environment must be modified to match your system.
 <hr/>
<p align="center"><sup>Copyright&copy; 2019 Xilinx</sup></p>
