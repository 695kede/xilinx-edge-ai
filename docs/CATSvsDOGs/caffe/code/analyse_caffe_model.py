#encoding: utf-8

# daniele.bagni@xilinx.com
# 32 Aug 2018

# ##################################################################################################
# USAGE
# python code/analyse_caffe_model.py -d ./models/alexnetBNnoLRN/m1/deploy_1_alexnetBNnoLRN.prototxt -w ./models/alexnetBNnoLRN/m1/snapshot_1_alexnetBNnoLRN__iter_12703.caffemodel

# ##################################################################################################



import os
import glob
import cv2
import caffe
import lmdb
import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")
import numpy as np
from numpy import prod, sum

from pprint import pprint

import argparse

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--description", required=True, help="description model")
ap.add_argument("-w", "--weights", required=True, help="weights caffemodel")
args = vars(ap.parse_args())

from caffe.proto import caffe_pb2

caffe.set_mode_gpu() 

# ##################################################################################################
#Size of images
IMAGE_WIDTH = 227
IMAGE_HEIGHT = 227

# ##################################################################################################
#Read model architecture and trained model's weights
caffe_description = args["description"]
caffe_model       = args["weights"]


# ##################################################################################################
net = caffe.Net(caffe_description, caffe_model, caffe.TEST)


'''

# ##################################################################################################
MEAN_FILE    = "/home/danieleb/ML/cats-vs-dogs/input/mean.binaryproto"

#Read mean image
mean_blob = caffe_pb2.BlobProto()
with open(MEAN_FILE) as f:
    mean_blob.ParseFromString(f.read())
mean_array = np.asarray(mean_blob.data, dtype=np.float32).reshape(
    (mean_blob.channels, mean_blob.height, mean_blob.width))

#Define image transformers
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape}) # input preprocessing: 'data' is the name of the input blob == net.inputs[0]
transformer.set_mean('data', mean_array)

transformer.set_transpose('data', (2,0,1))
#transformer.set_raw_scale('data', 255)         # use only with caffe.io.load_image()      
#transformer.set_channel_swap('data', (2,1,0))  # do not need to use it with OpenCV

# reshape the blobs so that they match the image shape. 
#net.blobs['data'].reshape(1,3,32,32)

# ##################################################################################################
'''


# from "Deep Learning tutorial on Caffe technology: basic commands, python and C++ code" by Christopher Bourez's blog

# The names of input layers of the net are given by print net.inputs

# The net contains two ordered dictionaries
# net.blobs for input data and its propagation in the layers :
# net.blobs['data'] contains input data, an array of shape (1, 1, 100, 100)
# net.blobs['conv'] contains computed data in layer ‘conv’ (1, 3, 96, 96) initialiazed with zeros.
#
# To print the infos,
print "\nINFO: layers and shapes:"
pprint( [(k, v.data.shape) for k, v in net.blobs.items()] )

# net.params a vector of blobs for weight and bias parameters
# net.params['conv'][0] contains the weight parameters, an array of shape (3, 1, 5, 5)
# net.params['conv'][1] contains the bias parameters, an array of shape (3,) initialiazed with ‘weight_filler’ and ‘bias_filler’ algorithms.
#
# To print the infos :
print "\nINFO: layers and parameters:"
pprint( [(k, v[0].data.shape, v[1].data.shape) for k, v in net.params.items()] )


#Blobs are memory abstraction objects (with execution depending on the mode), and data is contained in the field data as an array :
#print net.blobs['conv'].data.shape

# blob.data.shape can be used to find the shape of the different layers in your net. Loop across it to get shape of each layer.


print "INFO: for each layer, show the parameters:"
for layer_name, param in net.params.iteritems():
    print layer_name + '\t' + str(param[0].data.shape), str(param[1].data.shape)
print "\n"

print "INFO: for each layer, show the output shape:"
for layer_name, blob in net.blobs.iteritems():
    print layer_name + '\t' + str(blob.data.shape)
print "\n"	
	

print "\nTotal number of parameters: " + str(sum([prod(v[0].data.shape) for k, v in net.params.items()]))


'''
# generate dictionaries of the parameters

params = net.params.keys()
source_params = { pr: (net.params[pr][0].data, net.params[pr][1].data) for pr in params}

# Layer name, weights and bias from caffe model

for pr in params:
    print "Layer   = " + str(pr)

for pr in params:
    print "Weights = " + str(source_params[pr][0])

for pr in params:
    print "Bias    = " + str(source_params[pr][1])    

'''

'''    
print '\n Trainable net parameters (stored in net.params)'
net_params = net.params
pprint( [(key, net_params[key]) for key in net_params] )

print '\n Input data to the net (stored in net.blobs)'
net_blobs = net.blobs
pprint( [(key, net_blobs[key]) for key in net_blobs] )
'''

print '\n Net Layer Dictionary'
net_layer_dict = net.layer_dict
pprint( [(key, net_layer_dict[key]) for key in net_layer_dict] )






