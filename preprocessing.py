#!/usr/bin/env python
# coding: utf-8

# In[9]:


#-*-coding:utf-8-*-

# import libraries
import cv2 as cv
import os
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.image as img

from PIL import Image


# In[10]:


# base directory
base_dir = "/home/dmsai2/Desktop/DCC2022/data"
processed_dir = "/home/dmsai2/Desktop/DCC2022/pdata"


# In[11]:


# load class list from directory
cls_list = os.listdir(base_dir)
cls_list


# In[12]:


# change class name to integer
clsdict = {}
for i, cls in enumerate(cls_list):
    clsdict[i] = cls
clsdict


# In[13]:


numdata = 0
numclass = 0
data_list = {}

for cls in cls_list:
    imlist = os.listdir(base_dir + "/" + cls)
    print(cls, end=' : ')
    print(len(imlist))
    numdata = numdata + len(imlist)
    data_list[cls] = imlist
    print("-"*12)

numclass = len(cls_list)

print("num of class :", numclass)
print("num of data :", numdata)


# In[14]:


def cmyk2rgb(img):
    return img.convert("RGB")


# In[15]:


def preprocess(img_dir):
    
    img = Image.open(img_dir)
    
    if img.mode == "RGB":
        pass
    else:
        img = img.convert("RGB")
    
    img = np.array(img)
    
    # if the image is png type,
    # remove the "alpha channel"
    # and change type from "float" to "np.uint8"
    # if img_dir[-3:] == "png":
    #     img = (plt.imread(img_dir)[:,:,0:3]*255).astype(np.uint8)
    # else: # jpg
    #     img = plt.imread(img_dir)[:,:,0:3]
    #     shp = img.shape
    #     
    #     if len(shp) == 2: # single channel
    #         img = np.reshape(img, (shp[0], shp[1], 1))
    #         img = np.concatenate((img, img, img), axis=2)
    #         # print(img.shape)
        
    # get channel
    x, y, c = img.shape
    
    # print(img.shape)
    
    # top to bottom
    cnt = 0
    for i in range(x):
        if np.all(img[i,:,0]==255) and np.all(img[i,:,1]==255) and np.all(img[i,:,2]==255):
            continue
        else:
            cnt = i
            break
    
    # print(cnt)
    for _ in range(cnt):
        img = np.delete(img, 0, axis=0)
    #print(img.shape)
    
    x, y, c = img.shape
    
    # bottom to top
    cnt = 0
    for i in range(x):
        if np.all(img[x-i-1,:,0]==255) and np.all(img[x-i-1,:,1]==255) and np.all(img[x-i-1,:,2]==255):
            continue
        else:
            cnt = i
            break
    
    #print(cnt)
    for _ in range(cnt):
        img = np.delete(img, -1, axis=0)
    #print(img.shape)
    
    x, y, c = img.shape
    
    # left to right
    cnt = 0
    for j in range(y):
        if np.all(img[:,j,0]==255) and np.all(img[:,j,1]==255) and np.all(img[:,j,2]==255):
            continue
        else:
            cnt = j
            break
    
    #print(cnt)
    for _ in range(cnt):
        img = np.delete(img, 0, axis=1)
    #print(img.shape)
    
    x, y, c = img.shape
    
    # right to left
    cnt = 0
    for j in range(y):
        if np.all(img[:,y-j-1,0]==255) and np.all(img[:,y-j-1,1]==255) and np.all(img[:,y-j-1,2]==255):
            continue
        else:
            cnt = j
            break
    
    #print(cnt)
    for _ in range(cnt):
        img = np.delete(img, -1, axis=1)
    #print(img.shape)
    
    x, y, c = img.shape
    
    # plt.imshow(img)
    
    # make to squre shape
    if x == y:
        timg = img
    elif x > y:
        #print("X")
        padding1 = np.ones(shape=(x, (x - y) // 2, 3), dtype=np.int8)*255
        padding2 = np.ones(shape=(x, x - y - (x - y) // 2, 3), dtype=np.int8)*255
        #print(padding1.shape, padding2.shape)
        timg = np.concatenate((padding1, img, padding2), axis=1)
        #print(timg.shape)
    else: # x < y
        #print("Y")
        padding1 = np.ones(shape=((y - x) // 2, y, 3), dtype=np.int8)*255
        padding2 = np.ones(shape=(y - x - (y - x) // 2, y, 3), dtype=np.int8)*255
        
        #print(padding1.shape, padding2.shape)
        
        timg = np.concatenate((padding1, img, padding2), axis=0)
        
        #print(timg.shape)
    
    return timg


# In[16]:


# make directory in pdata folder
for label in data_list.keys():
    if os.path.isdir(processed_dir + "/" + label):
        print(processed_dir + "/" + label, "already exists")
    else:
        os.mkdir(processed_dir + "/" + label)
        print(processed_dir + "/" + label, "made")


# In[17]:


cnt = 0

error_list = []

# crop images and save data
for label in data_list.keys():
    print("LABEL :", label)
    for img_dir in data_list[label]:
        cnt = cnt + 1
        print(label, img_dir)
        
        if os.path.isfile(processed_dir + "/" + label + "/" + img_dir):
            print(f"file already exists! - {cnt}/{numdata} - {int((cnt / numdata * 100000))/1000}%")
        else:
            try:
                timg = preprocess(base_dir + "/" + label + "/" + img_dir)
                rimg = cv.resize(timg, dsize=(200, 200))
                rimg = Image.fromarray(np.uint8(rimg))
                img_name = img_dir[:-3] + "jpg"
                rimg.save(processed_dir + "/" + label + "/" + img_name)
                # cv.imwrite(processed_dir + "/" + label + "/" + img_dir, rimg)
                print(f"completed! - {cnt}/{numdata} - {int((cnt / numdata * 100000))/1000}%")
            except Exception as e:
                print(e)
                print("Error occured! at " + label + " : " + img_dir)
                error_list.append(label + " " + img_dir)
                
print("ERROR LIST")
print(error_list)


# In[59]:


# # grayscale images
# cnt = 0

# d = {1:0, 2:0, 3:0, 4:0}

# # crop images and save data
# for label in data_list.keys():
#     print("LABEL :", label)
#     for img_dir in data_list[label]:
#         cnt = cnt + 1
#         testimg = plt.imread(base_dir + "/" + label + "/" + img_dir)
#         sh = testimg.shape
            
#         if (len(sh) != 3) or (sh[2] != 3):
#             print(label, img_dir, sh)
#             cnt += 1
    
# print(cnt)

# cnt = 0

# dpng = {1:0, 2:0, 3:0, 4:0, 5:0, 6:0}
# djpg = {1:0, 2:0, 3:0, 4:0, 5:0, 6:0}

# # crop images and save data
# for label in data_list.keys():
#     print("LABEL :", label)
#     for img_dir in data_list[label]:
#         cnt = cnt + 1
#         testimg = plt.imread(base_dir + "/" + label + "/" + img_dir)
#         sh = testimg.shape
        
#         if img_dir[-3:] == "png":
#             if len(sh) == 2:
#                 dpng[1] = dpng[1] + 1
#             else:
#                 dpng[sh[2]] = dpng[sh[2]] + 1
#         else:
#             if len(sh) == 2:
#                 djpg[1] = djpg[1] + 1
#             else:
#                 djpg[sh[2]] = djpg[sh[2]] + 1
            
# print(dpng)
# print(djpg)


# In[61]:


# imgmodes = {}

# # crop images and save data
# for label in data_list.keys():
#     print("LABEL :", label)
#     for img_dir in data_list[label]:
#         timg = Image.open(base_dir + "/" + label + "/" + img_dir)
#         imgmode = timg.mode
        
#         if imgmode in imgmodes.keys():
#             imgmodes[imgmode] = imgmodes[imgmode] + 1
#         else:
#             imgmodes[imgmode] = 1
            
# print(imgmodes)


# In[ ]:




