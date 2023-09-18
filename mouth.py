#coding:utf8
import sys
import os
import time
import numpy as np
import cv2
import dlib

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision
from torchvision import datasets, models, transforms

# Define a CNN(convolutional neural network) model
class simpleconv3(nn.Module):
    def __init__(self,nclass):
        super(simpleconv3,self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, 2)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, 2)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3, 2)
        self.bn3 = nn.BatchNorm2d(128)
        self.fc1 = nn.Linear(128 * 5 * 5 , 1024)
        self.fc2 = nn.Linear(1024 , 128)
        self.fc3 = nn.Linear(128 , nclass)

    def forward(self , x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = x.view(-1 , 128 * 5 * 5) 
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
# Pre-trained models and predictors paths
PREDICTOR_PATH = "./models/shape_predictor_68_face_landmarks.dat"
predictor = dlib.shape_predictor(PREDICTOR_PATH)
cascade_path='./models/haarcascade_frontalface_default.xml'
cascade = cv2.CascadeClassifier(cascade_path)

# Extract face landmarks
def get_landmarks(im,rects):
    x,y,w,h =rects[0]
    rect=dlib.rectangle(int(x),int(y),int(x+w),int(y+h)) 
    return np.matrix([[p.x, p.y] for p in predictor(im, rect).parts()])

# Extract lip region from image using landmarks
def getlipfromimage(im, landmarks):
    xmin = 10000
    xmax = 0
    ymin = 10000
    ymax = 0

    # mouth outside（48-59）mouth inside（60-67）
    for i in range(48, 67):
        x = landmarks[i, 0]
        y = landmarks[i, 1]
        if x < xmin:
            xmin = x
        if x > xmax:
            xmax = x
        if y < ymin:
            ymin = y
        if y > ymax:
            ymax = y

    roiwidth = xmax - xmin
    roiheight = ymax - ymin

    roi = im[ymin:ymax, xmin:xmax, 0:3]

    if roiwidth > roiheight:
        dstlen = 1.5 * roiwidth
    else:
        dstlen = 1.5 * roiheight

    diff_xlen = dstlen - roiwidth
    diff_ylen = dstlen - roiheight

    newx = xmin
    newy = ymin

    imagerows, imagecols, channel = im.shape
    if newx >= diff_xlen / 2 and newx + roiwidth + diff_xlen / 2 < imagecols:
        newx = newx - diff_xlen / 2
    elif newx < diff_xlen / 2:
        newx = 0
    else:
        newx = imagecols - dstlen

    if newy >= diff_ylen / 2 and newy + roiheight + diff_ylen / 2 < imagerows:
        newy = newy - diff_ylen / 2
    elif newy < diff_ylen / 2:
        newy = 0
    else:
        newy = imagerows - dstlen

    roi = im[int(newy):int(newy) + int(dstlen), int(newx):int(newx) + int(dstlen), 0:3]
    return roi,int(newy),int(newx),dstlen

# Model path and normalization for data
modelpath = "./models/model.pt"
testsize = 48
torch.no_grad()


data_transforms =  transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])])

# Main class for mouth processing
class Mouth:
    def __init__(self):
        self.modelpath = modelpath
        self.testsize = testsize
        self.net = simpleconv3(4)
        self.net.eval()
        self.net.load_state_dict(torch.load(self.modelpath,map_location=lambda storage,loc: storage))
        self.data_transforms = data_transforms
    def process(self,im):
        try:
            rects = cascade.detectMultiScale(im, 1.3,5)
            landmarks = get_landmarks(im,rects)
            roi,offsety,offsetx,dstlen = getlipfromimage(im, landmarks)
            bbox = (offsetx, offsety, int(dstlen), int(dstlen))
            print(roi.shape)
            roi = cv2.cvtColor(roi,cv2.COLOR_BGR2RGB)
            inferenceimage = cv2.resize(roi,(self.testsize,self.testsize),interpolation=cv2.INTER_NEAREST)
            imgblob = self.data_transforms(inferenceimage).unsqueeze(0)
            ## activation function using softmax for multiclass classification
            predict = F.softmax(self.net(imgblob)).cpu().data.numpy().copy()
            print(predict)
            index = np.argmax(predict)
            return str(index), bbox
        except:
            return "-1", (0, 0, 0, 0)

# Mapping of detected expressions to labels
expressions_dict = {
    "0": "neutral",
    "1": "pouting",
    "2": "smile",
    "3": "open_mouth",
    "-1": "undetectable"
}

# Function to process mouth in image and save labeled output
def mouth(img_path):
    im = cv2.imread(img_path, 1)
    result, bbox = ms.process(im)
    x, y, w, h = bbox
    label = expressions_dict[result]
    
    # Drawing the rectangle
    cv2.rectangle(im, (x, y), (x+w, y+h), (0, 255, 0), 2)
    # Putting the label
    cv2.putText(im, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
    
    # Saving the image with rectangle and label
    output_path = os.path.join("output_images", os.path.basename(img_path))
    cv2.imwrite(output_path, im)
    
    return result

#processing images passed as command-line arguments
if __name__== '__main__':
    ms = Mouth()
    images = os.listdir(sys.argv[1])
    for image in images:
        img = cv2.imread(os.path.join(sys.argv[1],image))
        result = ms.process(img)
        print(result)
        
