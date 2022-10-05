#!/usr/bin/env python
# coding: utf-8

# In[26]:


import os
import numpy as np
import time

from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as img

import pandas as pd
import seaborn as sns

import csv

get_ipython().run_line_magic('matplotlib', 'inline')


# In[27]:


from sklearn.cluster import KMeans
import cv2


# In[28]:


# base directory
base_dir = "/home/dmsai2/Desktop/DCC2022/data"
processed_dir = "/home/dmsai2/Desktop/DCC2022/pdata"
sub_data_dir = "/home/dmsai2/Desktop/DCC2022/tdata"


# In[29]:


# load class list from directory
cls_list = os.listdir(base_dir)
print(*cls_list)


# In[30]:


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


# In[31]:


numsubcls = 0
subcls_list = {}

for cls in cls_list:
    sub_list = os.listdir(sub_data_dir + "/" + cls)
    subcls_list[cls] = sub_list
    
numsubcls = len(subcls_list)
print("num of subclass :", numsubcls)


# In[32]:


data = []
cnt = 0

if os.path.isfile('processed_df.csv'):
    print("reading data from csv.")
    df = pd.read_csv('processed_df.csv', index_col=0)
    df = df.astype({'img':'string', 'label':'string', 'ftype':'string', 'cmap':'string', 'fsize':'float', 'fmtime':'datetime64[ns]', 'width':'int', 'height':'int', 'channel':'int', 'subclass':'string', 'mode_rate':'float'})
    print(df.head())
    
else:
    print("generating data")
    for label in data_list.keys():
        for subcls in subcls_list[label]:
            subcls_path = f"{sub_data_dir}/{label}/{subcls}"
            imglist = os.listdir(subcls_path)

            for img_dir in imglist:
                img_path = f"{subcls_path}/{img_dir}"
                img = Image.open(img_path)
                imgarr = np.array(img)
                if len(imgarr.shape) == 2:
                    w, h = imgarr.shape
                    c = 1
                else:
                    w, h, c = imgarr.shape
                fsize = (os.path.getsize(img_path) / 1024.)
                fmtime = time.ctime(os.path.getmtime(img_path))
                data.append([img_dir[:-4], label, img_dir[-3:], img.mode, fsize, fmtime, w, h, c, subcls])
                cnt += 1
                print(f"{(int((cnt / numdata) * 100000.) / 1000.)}% - {cnt}/{numdata}")
    
    df = pd.DataFrame(data, columns=['img','label','ftype', 'cmap', 'fsize', 'fmtime', 'width', 'height', 'channel', 'subclass'])
    df = df.astype({'img':'string', 'label':'string', 'ftype':'string', 'cmap':'string', 'fsize':'float', 'fmtime':'datetime64[ns]', 'width':'int', 'height':'int', 'channel':'int', 'subclass':'string'})
    print(df.head())


# In[33]:


df.info()


# In[34]:


df.groupby('ftype')['cmap'].value_counts()


# In[35]:


def centroid_histogram(clt):
    # grab the number of different clusters and create a histogram
    # based on the number of pixels assigned to each cluster
    numLabels = np.arange(0, len(np.unique(clt.labels_)) + 1)
    (hist, _) = np.histogram(clt.labels_, bins=numLabels)

    # normalize the histogram, such that it sums to one
    hist = hist.astype("float")
    hist /= hist.sum()

    # return the histogram
    return hist

def plot_colors(hist, centroids):
    # initialize the bar chart representing the relative frequency
    # of each of the colors
    bar = np.zeros((50, 300, 3), dtype="uint8")
    startX = 0

    # loop over the percentage of each cluster and the color of
    # each cluster
    for (percent, color) in zip(hist, centroids):
        # plot the relative percentage of each cluster
        endX = startX + (percent * 300)
        cv2.rectangle(bar, (int(startX), 0), (int(endX), 50),
                      color.astype("uint8").tolist(), -1)
        startX = endX

    # return the bar chart
    return bar

def analyzeimg(img_dir : str, isprint=False):
    img = Image.open(img_dir)
    img = img.convert('RGB')

    img = np.array(img)
    w, h, c = img.shape

    rate = min(w, h) / 200.
    rw, rh = int(w / rate), int(h / rate)
    
    img = cv2.resize(img, dsize=(rw, rh))
    
    w, h, c = img.shape
    
    img = img.reshape((w * h, 3))
    
    if isprint: plt.imshow(img)
    
    k = 30
    clt = KMeans(n_clusters=k)
    clt.fit(img)
    
    if isprint:
        for center in clt.cluster_centers_:
            print("color :", center)
        
    hist = centroid_histogram(clt)
    if isprint: print("each % :", hist)
    
    bar = plot_colors(hist, clt.cluster_centers_)

    if isprint:
        # show our color bart
        plt.figure()
        # plt.axis("off")
        plt.imshow(bar)
        plt.show()
    
    avg = sum([clt.cluster_centers_[i] * hist[i] for i in range(k)])
    mode = clt.cluster_centers_[list(hist).index(max(hist))]
    if isprint: print(f"average : {avg}\nmode : {mode} {int(hist[list(hist).index(max(hist))]*100)}%")
    
    # average, mode, clusters(RGB), hist(each rate), rate of mode color
    return avg, mode, clt.cluster_centers_, hist, hist[list(hist).index(max(hist))]*100


# In[36]:


avg, mode, clusters, hist, mode_rate = analyzeimg(base_dir + "/L2_52/uehhbgclrdhrnrkuuenj.jpg", False)
mode_rate


# In[37]:


df.describe()


# In[ ]:


for i, row in df.iterrows():
    print(f"{i}/{numdata} {i/numdata*100}%", row['label'], row['img'], end=' ')
    avg, mode, clusters, hist, mode_rate = analyzeimg(f"{base_dir}/{row['label']}/{row['img']}.{row['ftype']}", isprint=False)
    df.loc[i, 'mode_rate'] = mode_rate
    print("mode rate :", mode_rate)
    
print("done")


# In[ ]:


df.to_csv('processed_df.csv', mode='w')


# In[ ]:




