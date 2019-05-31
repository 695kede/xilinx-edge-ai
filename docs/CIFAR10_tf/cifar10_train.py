######################################################
# CIFAR-10 example with miniVGGNet
# Mark Harvey
# April 2019
######################################################
import os
import sys
import shutil
import tensorflow as tf
import numpy as np
from netmodel.miniVGGNet import miniVGGNet


#####################################################
# Set up directories and files
#####################################################

SCRIPT_DIR = os.getcwd()

TRAIN_GRAPH = 'training_graph.pb'
INFER_GRAPH = 'inference_graph.pb'
CHKPT_FILE = 'float_model.ckpt'

CHKPT_DIR = os.path.join(SCRIPT_DIR, 'chkpts')
TB_LOG_DIR = os.path.join(SCRIPT_DIR, 'tb_logs')
CHKPT_PATH = os.path.join(CHKPT_DIR, CHKPT_FILE)


# create a directory for the TensorBoard data if it doesn't already exist
# delete it and recreate if it already exists
if (os.path.exists(TB_LOG_DIR)):
    shutil.rmtree(TB_LOG_DIR)
os.makedirs(TB_LOG_DIR)
print("Directory " , TB_LOG_DIR ,  "created ") 


# create a directory for the checkpoints if it doesn't already exist
# delete it and recreate if it already exists
if (os.path.exists(CHKPT_DIR)):
    shutil.rmtree(CHKPT_DIR)
os.makedirs(CHKPT_DIR)
print("Directory " , CHKPT_DIR ,  "created ")



#####################################################
# Dataset preparation
#####################################################
# CIFAR10 dataset has 60k images. Training set is 50k, test set is 10k.
# Each image is 32x32x8bits
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

# Scale image data from range 0:255 to range 0:1
# Also converts train & test data to float from uint8
x_train = (x_train/255.0).astype(np.float32)
x_test = (x_test/255.0).astype(np.float32)

# take 5000 images & labels from the train dataset to create a validation set
x_valid = x_train[45000:]
y_valid = y_train[45000:]

# train dataset reduced to 45000 images
x_train = x_train[:45000]
y_train = y_train[:45000]


# one-hot encode the labels
y_train = tf.keras.utils.to_categorical(y_train, num_classes=10)
y_test = tf.keras.utils.to_categorical(y_test, num_classes=10)
y_valid = tf.keras.utils.to_categorical(y_valid, num_classes=10)



#####################################################
# Hyperparameters
#####################################################
LEARN_RATE = 0.0001
EPOCHS = 100
BATCHSIZE = 50


# calculate total number of batches per epoch
total_batches = int(len(x_train)/BATCHSIZE)


#####################################################
# Create the Computational graph
#####################################################

# define placeholders for the input images, labels, training mode and droput rate
images_in = tf.placeholder(tf.float32, shape=[None,32,32,3], name='images_in')
labels = tf.placeholder(tf.int32, shape=[None,10], name='labels')
train = tf.placeholder_with_default(False, shape=None, name='train')
drop = tf.placeholder_with_default(0.0, shape=None, name='drop')


# build the network, input comes from the 'images_in' placeholder
# training mode and dropout rate are also driven by placeholders
logits = miniVGGNet(inputs=images_in, is_training=train, drop_rate=drop)


# softmax cross entropy loss function
# needs one-hot encoded labels
loss = tf.reduce_mean(tf.losses.softmax_cross_entropy(logits=logits, onehot_labels=labels))


# Adaptive Momentum optimizer - minimize the loss
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
optimizer = tf.train.AdamOptimizer(learning_rate=LEARN_RATE)
with tf.control_dependencies(update_ops):
    train_op = optimizer.minimize(loss)



# Check to see if the prediction matches the label
correct_prediction = tf.equal(tf.argmax(logits, 1, output_type=tf.int32), tf.argmax(labels, 1, output_type=tf.int32)  )

 # Calculate accuracy as mean of the correct predictions
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


# top 5 and top 1 accuracy
in_top5 = tf.nn.in_top_k(predictions=logits, targets=tf.argmax(labels, 1), k=5)
in_top1 = tf.nn.in_top_k(predictions=logits, targets=tf.argmax(labels, 1), k=1)
top5_acc = tf.reduce_mean(tf.cast(in_top5, tf.float32))
top1_acc = tf.reduce_mean(tf.cast(in_top1, tf.float32))


# TensorBoard data collection
tf.summary.scalar('cross_entropy_loss', loss)
tf.summary.scalar('accuracy', accuracy)
tf.summary.image('input_images', images_in)


# set up saver object
saver = tf.train.Saver()



#####################################################
# Run the graph in a Session
#####################################################
# Launch the graph
with tf.Session() as sess:

    sess.run(tf.initializers.global_variables())
    
    # TensorBoard writer
    writer = tf.summary.FileWriter(TB_LOG_DIR, sess.graph)
    tb_summary = tf.summary.merge_all()

    # Training phase with training data
    print ('******************************')
    print ('TRAINING STARTED..')
    print ('******************************\n')
    for epoch in range(EPOCHS):
        print ("Epoch", epoch+1, "/", EPOCHS)

        # process all batches
        for i in range(total_batches):
            
            # fetch a batch from training dataset
            x_train_batch, y_train_batch = x_train[i*BATCHSIZE:i*BATCHSIZE+BATCHSIZE], y_train[i*BATCHSIZE:i*BATCHSIZE+BATCHSIZE]

            # Display training accuracy every 100 batches
            if i % 100 == 0:
              acc = sess.run(accuracy, feed_dict={images_in: x_test, labels: y_test})
              print (' Step: {:4d}  Training accuracy: {:1.4f}'.format(i,acc))

            # Run graph for optimization  - i.e. do the training
            _, s = sess.run([train_op, tb_summary], feed_dict={images_in: x_train_batch, labels: y_train_batch, train: True, drop: 0.25})
            writer.add_summary(s, (epoch*total_batches + i))


    print("\nTRAINING FINISHED\n")
    print ('******************************')
    writer.flush()
    writer.close()


    # Validation phase with validation dataset
    # calculate top-1 and top-5 accuracy with 'unseen' data
    print ('******************************')
    print("VALIDATION")
    print ('******************************\n')
    t5_acc,t1_acc = sess.run([top5_acc,top1_acc], feed_dict={images_in: x_valid, labels: y_valid})
    print (' Top 1 accuracy with validation set: {:1.4f}'.format(t1_acc))
    print (' Top 5 accuracy with validation set: {:1.4f}'.format(t5_acc))

    # save post-training checkpoint
    # this saves all the parameters of the trained network
    save_path = saver.save(sess, os.path.join(CHKPT_DIR, CHKPT_FILE))
    print('\nSaved checkpoint to %s' % os.path.join(CHKPT_DIR,CHKPT_FILE))

    '''
    # optional - write out training graph
    tf.train.write_graph(sess.graph_def, CHKPT_DIR, TRAIN_GRAPH, as_text=False)
    print('Saved binary graphDef to %s' % os.path.join(CHKPT_DIR,TRAIN_GRAPH))
    '''

#####  SESSION ENDS HERE #############



#####################################################
# Write out a graph for inference
#####################################################
# we cannot use the training graph for deployment with DNNDK
# we need to create a new graph with is_training set to False to disable dropout & batch norm
# this new graph does not have any nodes associated with training (loss, optimizer, etc)

with tf.Graph().as_default():

  # define placeholders for the input data
  x_1 = tf.placeholder(tf.float32, shape=[None,32,32,3], name='images_in')

  # call the miniVGGNet function with is_training=False & dropout rate=0
  logits_1 = miniVGGNet(x_1, is_training=False, drop_rate=0.0)

  tf.train.write_graph(tf.get_default_graph().as_graph_def(), CHKPT_DIR, INFER_GRAPH, as_text=False)
  print('Saved binary inference graph to %s' % os.path.join(CHKPT_DIR,INFER_GRAPH))

print('\nRun `tensorboard --logdir=%s --port 6006 --host localhost` to see the results.' % TB_LOG_DIR)
print('\nFINISHED!')
print ('******************************')

