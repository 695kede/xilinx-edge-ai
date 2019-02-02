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

# modified by daniele.bagni@xilinx.com
# date 30 Aug 2018 2018

# ##################################################################################################
# USAGE
# python code/6_make_predictions.py -d ./models/alexnetBNnoLRN/m1/deploy_1_alexnetBNnoLRN.prototxt -w ./models/alexnetBNnoLRN/m1/snapshot_1_alexnetBNnoLRN__iter_12703.caffemodel

# it computes the prediction accuracy for the CNN trainined on CATS cvs DOGS by using 1000 JPEG 227x227x3 images in
# the test directory (not belonging to the trainining or validation LMDB datasets)

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

from config import cats_vs_dogs_config as config

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

# mean file for CATSvsDOGS training dataset
MEAN_FILE    = config.MEAN_FILE                         # i.e. "/home/ML/cats-vs-dogs/input/mean.binaryproto"
# test dataset
TEST_DATASET = config.PROJ_JPG_DIR + "/test/*.jpg" # i.e. "/home/ML/cats-vs-dogs/input/jpg/test/*.jpg"


# ##################################################################################################
'''

#Image processing helper function

def transform_img(img, img_width=IMAGE_WIDTH, img_height=IMAGE_HEIGHT):

    #Histogram Equalization
    #img[:, :, 0] = cv2.equalizeHist(img[:, :, 0])
    #img[:, :, 1] = cv2.equalizeHist(img[:, :, 1])
    #img[:, :, 2] = cv2.equalizeHist(img[:, :, 2])

    #Image Resizing
    img = cv2.resize(img, (img_width, img_height), interpolation = cv2.INTER_CUBIC)

    return img

'''
# ##################################################################################################

'''
#Read mean image
mean_blob = caffe_pb2.BlobProto()
with open(MEAN_FILE) as f:
    mean_blob.ParseFromString(f.read())
mean_array = np.asarray(mean_blob.data, dtype=np.float32).reshape(
    (mean_blob.channels, mean_blob.height, mean_blob.width))
'''

mean_array = np.zeros((3,IMAGE_WIDTH, IMAGE_HEIGHT)) 
ONE = np.ones((IMAGE_WIDTH, IMAGE_HEIGHT))
mean_array[0, :, :] = ONE*106
mean_array[1, :, :] = ONE*116
mean_array[2, :, :] = ONE*124


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
#net.blobs['data'].reshape(1,3,227,227)

# ##################################################################################################
'''
Making predictions
'''

#Reading image paths
test_img_paths = [img_path for img_path in glob.glob(TEST_DATASET)]

NUMEL = len(test_img_paths)

#Making predictions
test_ids = np.zeros(([NUMEL,1]))
preds = np.zeros(([NUMEL, 2]))
idx = 0

for img_path in test_img_paths:

    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    #img = transform_img(img, img_width=IMAGE_WIDTH, img_height=IMAGE_HEIGHT) #DB: images do not need resizing 
    #cv2.imshow('img_path', img)
    #cv2.waitKey(0)    
    #img = caffe.io.load_image(img_path) # alternative way
    
    net.blobs['data'].data[...] = transformer.preprocess('data', img)
    out = net.forward()
    #best_n = net.blobs['prob'].data[0].flatten().argsort()[-1: -2:-1]
    #print("DBG INFO: ", best_n)
    pred_probas = out['prob'] # returns the probabilities of the 2 classes

    # compute top-5: take the last 5 elements [-5:] and reverse them [::-1]
    top5 = pred_probas.argsort()[-2:][::-1]

    filename = img_path.split("/jpg/test/")[1]

    '''    
    if '/jpg/val/cat/' in img_path:
        filename = img_path.split("/jpg/val/cat/")[1]
    elif '/jpg/val/dog/' in img_path:
        filename = img_path.split("/jpg/val/dog/")[1]
    else: # other
        print 'ERROR: your path name does not contain "/jpg/val/" '
        sys.exit(0)            
    '''
    
    if 'cat' in filename:
        label = 0
    elif 'dog' in filename:
        label = 1
    else:
        label = -1 # non existing

    test_ids[idx] = label
    preds[idx] = pred_probas
    #print("DBG INFO ", pred_probas)

    print("IMAGE: " + img_path)
    print("PREDICTED: %d" % preds[idx].argmax())
    print("EXPECTED : %d" % test_ids[idx])
    print '-------'

    idx = idx+1
    #if idx==100 :
    #   break

          

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
labelNames = ["cat", "dog"]

report=classification_report(testY, preds.argmax(axis=1), target_names=labelNames)
print(report)

from sklearn.metrics import accuracy_score
print('SKLEARN Accuracy = %.2f' % accuracy_score(testY, preds.argmax(axis=1)) )

# ##################################################################################################
# CHECK MANUALLY THE ACCURACY: false and positive predictions

list_predictions = np.array(preds.argmax(axis=1)) # actual predictions
list_str_num = np.array(testY)    # ground truth

tot_true  = 0
tot_false = 0
cat_true  = 0
cat_false = 0
dog_true  = 0
dog_false = 0

for ii in range(0, NUMEL) :
    n1 = list_str_num[ii]
    n2 = list_predictions[ii]
    diff = n1 - n2
    if diff == 0 :
        tot_true = tot_true + 1
        if n1==0: #cat
            cat_true = cat_true + 1
        elif n1==1: #dog
            dog_true = dog_true + 1            
    else:
        tot_false = tot_false+1
        if n1==0: #cat
            dog_false = dog_false + 1 #we predicted a "dog" but it was a "cat"
        elif n1==1: #dog
            cat_false = cat_false + 1 #we predicted a "cat" but it was a "dog"                    

print("\n")
print 'TOTAL NUMBER OF TRUE  PREDICTIONS = ', tot_true
print 'TOTAL NUMBER OF FALSE PREDICTIONS = ', tot_false
print 'TOTAL NUMBER OF true  dog PREDICTIONS = ', dog_true
print 'TOTAL NUMBER OF true  cat PREDICTIONS = ', cat_true
print 'TOTAL NUMBER OF cat predicted as dog  = ', dog_false
print 'TOTAL NUMBER OF dog predicted as cat  = ', cat_false


if (tot_true+tot_false) != NUMEL :
        print 'ERROR: number of total false and positive is not equal to the number of processed images'

recall =  float(tot_true)/(tot_true+tot_false)
print('MANUALLY COMPUTED RECALL = %.2f\n' % recall)

