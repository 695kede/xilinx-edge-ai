'''
Title           :make_predictions_1.py
Description     :This script makes predictions using the 1st trained model and generates a submission file.
Author          :Adil Moujahid
Date Created    :20160623
Date Modified   :20160625
version         :0.2
usage           :python make_predictions_1.py
python_version  :2.7.11
'''

# highly modified by daniele.bagni@xilinx.com
# date 19 September 2018

# ##################################################################################################
# USAGE
# python ./code/6_make_predictions.py -d ./models/miniVggNet/m3/deploy_3_miniVggNet.prototxt -w ./models/miniVggNet/m3/snapshot_3_miniVggNet__iter_40000.caffemodel

# it computes the prediction accuracy for the CNN trainined on CIFAR10 by using a 1000 JPEG images in
# the test directory (not belonging to the trainining or validation LMDB datasets)

# ##################################################################################################

import os
import glob
import sys
#sys.path.append('/home/danieleb/.virtualenvs/caffe_py27/local/lib/python2.7/site-packages/')
import cv2
import caffe
import lmdb
import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")
import numpy as np

from config import cifar10_config as config

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
IMAGE_WIDTH = 32
IMAGE_HEIGHT = 32

# mean file for CIFAR10 training dataset
MEAN_FILE    = config.MEAN_FILE                         # i.e. "/home/ML/cifar10/input/mean.binaryproto"
# test dataset
TEST_DATASET = config.CIFAR10_JPG_DIR + "/test/*.jpg" # i.e. "/home/ML/cifar10/input/cifar10_jpg/test/*.jpg"


# ##################################################################################################

'''
Image processing helper function
'''

def transform_img(img, img_width=IMAGE_WIDTH, img_height=IMAGE_HEIGHT):

    #Histogram Equalization
    #img[:, :, 0] = cv2.equalizeHist(img[:, :, 0])
    #img[:, :, 1] = cv2.equalizeHist(img[:, :, 1])
    #img[:, :, 2] = cv2.equalizeHist(img[:, :, 2])

    #Image Resizing
    img = cv2.resize(img, (img_width, img_height), interpolation = cv2.INTER_CUBIC)

    return img

# ##################################################################################################

'''
Reading mean image, caffe model and its weights 
'''

#Read mean image
mean_blob = caffe_pb2.BlobProto()
with open(MEAN_FILE) as f:
    mean_blob.ParseFromString(f.read())
mean_array = np.asarray(mean_blob.data, dtype=np.float32).reshape(
    (mean_blob.channels, mean_blob.height, mean_blob.width))


#Read model architecture and trained model's weights
caffe_description = args["description"]
caffe_model       = args["weights"]


net = caffe.Net(caffe_description, caffe_model, caffe.TEST)
#Define image transformers
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape}) # input preprocessing: 'data' is the name of the input blob == net.inputs[0]
transformer.set_mean('data', mean_array)


'''
- The set_transpose transforms an image from (256,256,3) to (3,256,256).
- The set_channel_swap function will change the channel ordering. Caffe uses BGR image format, so we need to 
  change the image from RGB to BGR. If you are using OpenCV to load the image, then this step is not necessary 
  since OpenCV also uses the BGR format
- The set_raw_scale is needed only if you load images with caffe.io.load_image(). You do not need it if using OpenCV.
  It means that the reference model operates on images in [0,255] range instead of [0,1]. 
'''

transformer.set_transpose('data', (2,0,1))
#transformer.set_raw_scale('data', 255)         # use only with caffe.io.load_image()      
#transformer.set_channel_swap('data', (2,1,0))  # do not need to use it with OpenCV

# reshape the blobs so that they match the image shape. 
#net.blobs['data'].reshape(1,3,32,32)

# ##################################################################################################
'''
Making predictions
'''

#Reading image paths
test_img_paths = [img_path for img_path in glob.glob(TEST_DATASET)]

NUMEL = len(test_img_paths)

#Making predictions
test_ids = np.zeros(([NUMEL,1]))
preds = np.zeros(([NUMEL, 10]))
idx = 0

tot_true  = 0
tot_false = 0
top5_true = 0
top5_false= 0

for img_path in test_img_paths:

    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    #img = transform_img(img, img_width=IMAGE_WIDTH, img_height=IMAGE_HEIGHT)
    #cv2.imshow('img_path', img)
    #cv2.waitKey(0)    
    #img = caffe.io.load_image(img_path) # alternative way
    
    net.blobs['data'].data[...] = transformer.preprocess('data', img)
    out = net.forward()
    best_n = net.blobs['prob'].data[0].flatten().argsort()[-1: -6:-1]
    #print("DBG INFO: ", best_n)
    pred_probas = out['prob'] # returns the probabilities of the 10 classes

    # compute top-5: take the last 5 elements [-5:] and reverse them [::-1]
    top5 = pred_probas.argsort()[-5:][::-1]

    if 'airplane' in img_path:
        label = 0
    elif 'automobile' in img_path:
        label = 1
    elif 'bird' in img_path:
        label = 2
    elif 'cat' in img_path:
        label = 3
    elif 'deer' in img_path:
        label = 4
    elif 'dog' in img_path:
        label = 5
    elif 'frog' in img_path:
        label = 6
    elif 'horse' in img_path:
        label = 7
    elif 'ship' in img_path:
        label = 8
    elif 'truck' in img_path:
        label = 9
    else:
        label = -1 # non existing

    if label in top5 :
        top5_true = top5_true + 1
    else :
        top5_false = top5_false + 1
    #print "DBG INFO ", label, top5

    test_ids[idx] = label
    preds[idx] = pred_probas
    #print("DBG INFO ", pred_probas)

    print("IMAGE: " + img_path)
    print("PREDICTED: %d" % preds[idx].argmax())
    print("EXPECTED : %d" % test_ids[idx])
    print '-------'

    idx = idx+1
#    if idx==100 :
#          break

          

# ##################################################################################################
# SKLEARN REPORT
'''
precision = tp / (tp+fp) = ability of the classifier to not label as positive a sample that is negative
recall    = tp / (tp+fn) = ability of the classifier to find all positive samples
F1-score  = weighter harmonic mean of precision and recall. Best value approaches 1 and worst 0
support   = number of occurrences
'''    


from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
lb     = LabelBinarizer()
testY  = lb.fit_transform(test_ids)
labelNames = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

report=classification_report(testY.argmax(axis=1), preds.argmax(axis=1), target_names=labelNames)
print(report)

from sklearn.metrics import accuracy_score
print('SKLEARN Accuracy = %.2f' % accuracy_score(testY.argmax(axis=1), preds.argmax(axis=1)) )

# ##################################################################################################
# CHECK MANUALLY THE ACCURACY: false and positive predictions

list_predictions = np.array(preds.argmax(axis=1)) # actual predictions
list_str_num = np.array(testY.argmax(axis=1))    # ground truth

for ii in range(0, NUMEL) :
    n1 = list_str_num[ii]
    n2 = list_predictions[ii]
    diff = n1 - n2
    if diff == 0 :
        tot_true = tot_true + 1
    else:
        tot_false = tot_false+1

top5_accuracy = float(top5_true) / (top5_true + top5_false)
print("\n")
print('TOP-5 ACCURACY                    = %.2f ' % top5_accuracy)
print 'TOP-5 FALSE                       = ', top5_false
print 'TOP-5 TRUE                        = ', top5_true
print("\n")
print 'TOTAL NUMBER OF TRUE  PREDICTIONS = ', tot_true
print 'TOTAL NUMBER OF FALSE PREDICTIONS = ', tot_false

if (tot_true+tot_false) != NUMEL :
        print 'ERROR: number of total false and positive is not equal to the number of processed images'
if (top5_true+top5_false) != NUMEL :
        print 'ERROR: number of top5 total false and positive is not equal to the number of processed images'        

recall =  float(tot_true)/(tot_true+tot_false)
print('MANUALLY COMPUTED RECALL = %.2f ' % recall)

