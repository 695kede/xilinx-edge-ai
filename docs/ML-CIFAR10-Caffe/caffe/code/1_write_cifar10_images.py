# USAGE
# python 1_write_cifar10_images.py
# (optional) --pathname /home/ML/cifar10/input/cifar10_jpg

# It downloads the CIFAR-10 dataset from KERAS library and put it into JPG images organized in 3 folders:
# train (50000 images) validation (9000 images) and test (1000 images) with their proper labels txt files.
#
# It also builds a 4th directory for Calibration during the Qunatization process with DeePhi DECENT tool.

# by daniele.bagni@xilinx.com

# ##################################################################################################

# set the matplotlib backend before any other backend, so that figures can be saved in the background
import matplotlib     
matplotlib.use("Agg")

import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")
import numpy as np

# import the necessary packages
from config import cifar10_config as config
from keras.datasets import cifar10
from datetime import datetime
import matplotlib.pyplot as plt 
import cv2
import os
import argparse

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--pathname", default=config.CIFAR10_JPG_DIR, help="path to the dataset")
args = vars(ap.parse_args())


path_root = args["pathname"] # root path name of dataset

if (not os.path.exists(path_root)): # create "path_root" directory if it does not exist
    os.mkdir(path_root)


# ##################################################################################################

# load the training and testing data from CIFAR-10 dataset, then scale it into the range [0, 1]
print("[INFO] loading CIFAR-10 data...")
((trainX, trainY ), (testX, testY )) = cifar10.load_data()
trainX = trainX.astype("float") / 255.0
testX = testX.astype("float") / 255.0

# initialize the label names for the CIFAR-10 dataset
labelNames = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

# ##################################################################################################
# BUILD THE VALIDATION SET with 9000 images

wrk_dir = path_root + "/val"

if (not os.path.exists(wrk_dir)): # create "val" directory if it does not exist
    os.mkdir(wrk_dir)

f_test = open(wrk_dir+"/test.txt", "w")   #open file test.txt"
f_lab  = open(wrk_dir+"/labels.txt", "w") #open file labels.txt"
for s in [0,1,2,3,4,5,6,7,8,9]:
    string = "%s\n" % labelNames[s] 
    f_lab.write(string) 
f_lab.close()
    
counter = [0,0,0,0,0,0,0,0,0,0]

a = np.arange(0, 10000)

val_count = 0
for i in a : 
    image = testX[int(i)]
    image2 = image *  255.0
    image2 = image2.astype("int")
    

    counter[ int(testY[int(i)]) ] =     counter[ int(testY[int(i)]) ] +1;

    if counter[ int(testY[int(i)]) ] <= 100 : #skip the first 100 images of each class
        continue

    val_count = val_count + 1    
    string = "%05d" % counter[ int(testY[int(i)]) ]

    class_name = labelNames[int(testY[int(i)])]

    path_name = wrk_dir + "/" + class_name 

    if (not os.path.exists(path_name)): # create directory if it does not exist
        os.mkdir(path_name) #https://github.com/BVLC/caffe/issues/3698

    path_name = wrk_dir + "/" + class_name + "/" + class_name + "_" + string + ".jpg"

    string = " %1d" % int(testY[int(i)]) 
    f_test.write(path_name + string + "\n")

    cv2.imwrite(path_name, image2)
    
    print(path_name)

f_test.close()


# ##################################################################################################
# BUILD THE TEST SET with 1000 images


wrk_dir = path_root + "/test"

if (not os.path.exists(wrk_dir)): # create "test" directory if it does not exist
    os.mkdir(wrk_dir)

f_test  = open(wrk_dir+"/test.txt", "w")   #open file test.txt"
f_test2 = open(wrk_dir+"/test2.txt", "w")   #open file test.txt"
f_lab  = open(wrk_dir+"/labels.txt", "w") #open file labels.txt"
for s in [0,1,2,3,4,5,6,7,8,9]:
    string = "%s\n" % labelNames[s] 
    f_lab.write(string) 
f_lab.close()
    
counter = [0,0,0,0,0,0,0,0,0,0]

a = np.arange(0, 10000)

test_count = 0
test2_count = -1

for i in a : 
    image = testX[int(i)]
    #image2= cv2.resize(image, (32,32), interpolation=cv2.INTER_AREA)
    #image2 = image2 *  255.0
    image2 = image *  255.0
    image2 = image2.astype("int")


    counter[ int(testY[int(i)]) ] =     counter[ int(testY[int(i)]) ] +1;

    if counter[ int(testY[int(i)]) ] >100: #take only the first 100 images per each class
        continue

    test_count = test_count +1
    test2_count = test2_count +1
    string = "%05d" % counter[ int(testY[int(i)]) ]

    class_name = labelNames[int(testY[int(i)])]

    '''
    path_name = wrk_dir + "/" + class_name 

    if (not os.path.exists(path_name)): # create directory if it does not exist
        os.mkdir(path_name) #https://github.com/BVLC/caffe/issues/3698

    path_name = wrk_dir + "/" + class_name + "/" + class_name + "_" + string + ".jpg"

    string2 = " %1d" % test2_count
    f_test.write(path_name + string2 + "\n")
    f_test2.write(class_name + "/" + class_name + "_" + string + ".jpg" + string2 + "\n")
    '''
    
    path_name = wrk_dir + "/" + class_name + "_" + string + ".jpg"

    string2 = " %1d" % test2_count
    f_test.write(path_name + string2 + "\n")
    f_test2.write(class_name + "_" + string + ".jpg" + string2 + "\n")



    
    cv2.imwrite(path_name, image2)
    #cv2.imshow(labelNames[int(testY[int(i)])], image2)
    #cv2.waitKey(0)
    
    print(path_name)

f_test.close()
f_test2.close()


# ##################################################################################################
# BUILD THE TRAIN IMAGES SET


wrk_dir = path_root + "/train"

if (not os.path.exists(wrk_dir)): # create "train" directory if it does not exist
    os.mkdir(wrk_dir)

f_test = open(wrk_dir + "/train.txt", "w")   #open file test.txt"
f_lab  = open(wrk_dir + "/labels.txt", "w") #open file labels.txt"
for s in [0,1,2,3,4,5,6,7,8,9]:
    string = "%s\n" % labelNames[s] 
    f_lab.write(string) 
f_lab.close()
    
counter = [0,0,0,0,0,0,0,0,0,0]

a = np.arange(0, 50000)

for i in a : 
    image = trainX[int(i)]
    #image2= cv2.resize(image, (32,32), interpolation=cv2.INTER_AREA)
    #image2 = image2 *  255.0
    image2 = image *  255.0
    image2 = image2.astype("int")
   
    counter[ int(trainY[int(i)]) ] =     counter[ int(trainY[int(i)]) ] +1;
    string = "%05d" % counter[ int(trainY[int(i)]) ]

    class_name = labelNames[int(trainY[int(i)])]

    path_name = wrk_dir + "/" + class_name 

    if (not os.path.exists(path_name)): # create directory if it does not exist
        os.mkdir(path_name)

    path_name = wrk_dir + "/" + class_name + "/" + class_name + "_" + string + ".jpg"

    string = " %1d" % int(trainY[int(i)]) 
    f_test.write(path_name + string + "\n")

    cv2.imwrite(path_name, image2)
    #cv2.imshow(labelNames[int(testY[int(i)])], image2)
    #cv2.waitKey(0)
    
    #print(path_name)

f_test.close()

# ##################################################################################################
# BUILD THE CALIBRATION IMAGES SET


wrk_dir = path_root + "/calib"

if (not os.path.exists(wrk_dir)): # create "calibration" directory if it does not exist
    os.mkdir(wrk_dir)

f_calib = open(wrk_dir + "/calibration.txt", "w")   #open file calibration.txt"
for s in [0,1,2,3,4,5,6,7,8,9]:
    string = "%s\n" % labelNames[s]
    
counter = [0,0,0,0,0,0,0,0,0,0]

a = np.arange(0, 50000)

calib_count = -1
for i in a : 
    image = trainX[int(i)]
    #image2= cv2.resize(image, (32,32), interpolation=cv2.INTER_AREA)
    #image2 = image2 *  255.0
    image2 = image *  255.0
    image2 = image2.astype("int")
   
    counter[ int(trainY[int(i)]) ] =     counter[ int(trainY[int(i)]) ] +1;

    if counter[ int(trainY[int(i)]) ] > 100 : #take only the first 100 images per each class
        continue

    calib_count = calib_count + 1
    string = "%05d" % counter[ int(trainY[int(i)]) ]

    class_name = labelNames[int(trainY[int(i)])]

    path_name = wrk_dir + "/" + class_name 

    if (not os.path.exists(path_name)): # create directory if it does not exist
        os.mkdir(path_name)

    path_name = wrk_dir + "/" + class_name + "/" + class_name + "_" + string + ".jpg"

    string2 = " %1d" % int(calib_count) 
    f_calib.write(class_name + "/" + class_name + "_" + string + ".jpg" + string2 + "\n")

    cv2.imwrite(path_name, image2)
    #cv2.imshow(labelNames[int(testY[int(i)])], image2)
    #cv2.waitKey(0)
    
    #print(path_name)

f_calib.close()

print "Test       set contains ", test_count,  " images"
print "Validation set contains ", val_count,   " images"
print "Calibrationset contains ", calib_count+1, " images"
print("END\n")

