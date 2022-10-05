#-*-coding:utf-8-*-

# import libraries
import cv2 as cv
import os
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.image as img

# base directory
base_dir = "/home/dmsai2/Desktop/DCC2022/data"

# load class list from directory
cls_list = os.listdir(base_dir)

# change class name to integer
clsdict = {}
for i, cls in enumerate(cls_list):
    clsdict[i] = cls

cnt = 0
data_list = {}

for cls in cls_list:
    imlist = os.listdir(base_dir + "/" + cls)
    print(cls, end=' : ')
    print(len(imlist))
    cnt = cnt + len(imlist)
    data_list[cls] = imlist
    print("-"*12)

print("num of class :", len(cls_list))
print("num of data :", cnt)

# load images and labels
images = []
labels = []

for label in data_list.keys():
    print(label)
    for imgdir in data_list[label]:
        images.append(img.imread(base_dir + "/" + label + "/" + imgdir))
        labels.append(label)

print("load done")
