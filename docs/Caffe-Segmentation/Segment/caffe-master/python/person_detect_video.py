#coding=utf_8
import numpy as np
import cv2
import scipy.misc
import scipy.io
import os
import sys
import math
reload(sys)
sys.setdefaultencoding('utf-8')

caffe_root = "/home/liuji/caffe/"
os.chdir(caffe_root + 'python')

from PIL import Image,ImageDraw,ImageFont
font = ImageFont.truetype('/home/liuji/caffe_old_180725/examples/shuer_ssd/ssd_prototxt/simsun.ttc',40)
import caffe
caffe.set_device(0)
caffe.set_mode_gpu()

from google.protobuf import text_format
from caffe.proto import caffe_pb2
from google.protobuf import text_format
from caffe.proto import caffe_pb2
import time

# load plate  labels
labelmap_file = '/home/liuji/caffe_old_180725/examples/shuer_ssd/ssd_prototxt/models/labelmap_person.prototxt'
file = open(labelmap_file, 'r')
labelmap = caffe_pb2.LabelMap()
text_format.Merge(str(file.read()), labelmap)

def get_labelname(labelmap, labels):
  num_labels = len(labelmap.item)
  labelnames = []
  if type(labels) is not list:
    labels = [labels]
  for label in labels:
    found = False
    for i in xrange(0, num_labels):
      if label == labelmap.item[i].label:
        found = True
        labelnames.append(labelmap.item[i].display_name)
        break
    assert found == True
  return labelnames

ssd_model_def = '/home/liuji/caffe/examples/refineDet/refindet_compress/ST20180822G01/deploy.prototxt'
ssd_model_weights= '/home/liuji/caffe/examples/refineDet/refindet_compress/ST20180822G01/deploy.caffemodel'
#ssd_model_def = './ssd_deploy/deploy_f.prototxt'
#ssd_model_weights = './ssd_deploy/SSD_480x360_random_lr_f.caffemodel'
font = cv2.FONT_HERSHEY_SIMPLEX
ssd_net = caffe.Net(ssd_model_def,      # defines the structure of the model
                    ssd_model_weights,  # contains the trained weights
                    caffe.TEST)         # use test mode (e.g., don't perform dropout)
# input preprocessing: 'data' is the name of the input blob == net.inputs[0]
ssd_transformer = caffe.io.Transformer({'data': ssd_net.blobs['data'].data.shape})
ssd_transformer.set_transpose('data', (2, 0, 1))
ssd_transformer.set_mean('data', np.array([104,117,123])) # mean pixel
# set net to batch size of 1
image_resize_height = 480
image_resize_width = 640
#font=cv2.FONT_HERSHEY_SIMPLEX

def detectOnePic(image_ori):
  global frame
  height, width = image_ori.shape[0:2]
  image_resize = cv2.resize(image_ori, (image_resize_width, image_resize_height), interpolation=cv2.INTER_AREA)
  ssd_net.blobs['data'].reshape(1,3,image_resize_height,image_resize_width)
  transformed_image = ssd_transformer.preprocess('data',image_resize)
  ssd_net.blobs['data'].data[...] = transformed_image
  start = time.clock()
  detections = ssd_net.forward()['detection_out']
  end = time.clock()
  det_label = detections[0,0,:,1]
  det_conf = detections[0,0,:,2]
  det_xmin = detections[0,0,:,3]
  det_ymin = detections[0,0,:,4]
  det_xmax = detections[0,0,:,5]
  det_ymax = detections[0,0,:,6]
  # Get detections with confidence higher than 0.6.
  top_indices = [i for i, conf in enumerate(det_conf) if conf >= 0.05]
  top_conf = det_conf[top_indices]
  top_label_indices = det_label[top_indices].tolist()
  top_labels = get_labelname(labelmap, top_label_indices)
  top_xmin = det_xmin[top_indices]
  top_ymin = det_ymin[top_indices]
  top_xmax = det_xmax[top_indices]
  top_ymax = det_ymax[top_indices]

  if top_conf.size > 0:    
    size = len(top_labels)
    print "size:", size
    for i in range(size):
      xmin = (int)(width * top_xmin[i])
      ymin = (int)(height * top_ymin[i])
      xmax = (int)(width * top_xmax[i])
      ymax = (int)(height * top_ymax[i])
      
      if top_labels[i] == "person" and top_conf[i] >= 0.6:     
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (255, 255, 0), 2)
        cv2.putText(frame, 'person', (xmin, ymin + 10), font, 0.4, (255, 255, 255), 1) 
        cv2.putText(frame, str(top_conf[i]), (xmin, ymin - 10), font, 0.4, (255, 255, 255), 1)
        #fp.writelines(image_name[:-4] + " " + str(xmin) + " " + str(ymin) + " " + str(xmax - xmin) + " " + str(ymax - ymin) + " " + str(top_conf[i]) + "\n")
  return frame

image_dir = "/home/liuji/img/"
#fp = open("/home/liuji/wenjing/img.txt", "w")
image_names = os.listdir(image_dir)
for image_name in image_names:
  frame = cv2.imread(image_dir + image_name)  
  if frame is None:
    continue 
  frame = detectOnePic(frame)
  cv2.imwrite("/home/liuji/img_res/" + image_name, frame)
#fp.close()

'''
frame = cv2.imread("500470600.jpg")
frame = detectOnePic(frame)
cv2.imwrite("500470600_out.jpg", frame)
'''
 
#video_name_list = os.listdir("/home/liuji/人形检测视频/")
#for video_name in video_name_list:
#  if video_name == ".DS_Store":
#    continue
#  videoCapture = cv2.VideoCapture("/home/liuji/人形检测视频/" + video_name)
#  fps = videoCapture.get(cv2.cv.CV_CAP_PROP_FPS)
#  size = (int(videoCapture.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)),
#        int(videoCapture.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT)))
#  success, frame = videoCapture.read()
#  videoWriter = cv2.VideoWriter('./人形检测视频/' + video_name.split(".")[0] + ".avi", cv2.cv.CV_FOURCC('M', 'P', '4', '2'), 25, size)
#  frame_id=0
#
#  while success:
#    frame_id=frame_id+1
#    print frame_id
#    if (frame_id%1500)==0:
#      print "%d mins: "%(frame_id/1500)
#    frame = detectOnePic(frame)
#    videoWriter.write(frame)
#    cv2.waitKey(1)
#    success, frame = videoCapture.read()
#   
