import os



currentDir = os.getcwd()
print currentDir

os.chdir('/home/dschreib/caffe_segnet')

import sys
sys.path.insert(0, './python')
import matplotlib
matplotlib.use('Agg')
import caffe

import time
import numpy as np
import json
import scipy
import argparse
import math
import pylab
from sklearn.preprocessing import normalize
import cv2

args = {}
args['model'] = '/home/dschreib/Corals/Coral_Caffe_SegNet/Models/segnet_inference_coral.prototxt'
args['weights'] = '/home/dschreib/Corals/Coral_Caffe_SegNet/Models/Inference/coral_weights_05_22_3000_new.caffemodel'
args['iter'] = '203'

print int(args['iter'])

caffe.set_device(1)
caffe.set_mode_gpu()

net = caffe.Net(args['model'],
                args['weights'],
                caffe.TEST)
#net.set_device(1)


net.forward()


image = net.blobs['data'].data
label = net.blobs['label'].data
predicted = net.blobs['prob'].data
image = np.squeeze(image[0,:,:,:])
output = np.squeeze(predicted[0,:,:,:])
ind = np.argmax(output, axis=0)

r = ind.copy()
g = ind.copy()
b = ind.copy()
r_gt = label.copy()
g_gt = label.copy()
b_gt = label.copy()

cv2.imwrite('temp0.png', r)
cv2.imwrite('temp1.png', np.squeeze(label))
cv2.imwrite('temp2.png', np.swapaxes(np.transpose(image),0,1))

#print type(image)
#print np.swapaxes(np.transpose(image),0,1).shape
#print np.squeeze(label).shape
#print ind.shape

print (np.squeeze(label) == r).sum(axis = 0).sum(axis = 0) / (640.0 * 480.0)

accuracy = []


print time.time()

for i in range(0, int(args['iter'])):

    net.forward()
    image = net.blobs['data'].data
    label = net.blobs['label'].data
    predicted = net.blobs['prob'].data
    image = np.squeeze(image[0,:,:,:])
    output = np.squeeze(predicted[0,:,:,:])
    ind = np.argmax(output, axis=0)

    r = ind.copy()
    g = ind.copy()
    b = ind.copy()
    r_gt = label.copy()
    g_gt = label.copy()
    b_gt = label.copy()

    cv2.imwrite('test_result_' + str(i) + '.png', r)
    cv2.imwrite('test_label_' + str(i) + '.png', np.squeeze(label))
    cv2.imwrite('test_image_' + str(i) + '.png', np.swapaxes(np.transpose(image),0,1))
    accuracy.append((np.squeeze(label) == r).sum(axis=0).sum(axis=0)/(640.0 * 480.0))
print output.shape
print np.asarray(accuracy).sum(axis=0) / float(args['iter'])
print os.getcwd()
print time.time()
