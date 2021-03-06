# -*- coding: utf-8 -*-
"""Untitled8.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1GvQN8kWCc7L8lRWTpRciG6BeuxEofRI8
"""

from google.colab import drive

from google.colab import drive
drive.mount('/content/drive')

drive.mount('/content/drive')

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

meta_csv=pd.read_csv('/content/drive/MyDrive/DataScience/Meta.csv')   #Read the file
test_csv=pd.read_csv('/content/drive/MyDrive/DataScience/Test.csv')    #Read the file
train_csv=pd.read_csv('/content/drive/MyDrive/DataScience/Train.csv')   #Read the file

meta_csv.dropna(axis=0,inplace=True)
train_csv.dropna(axis=0,inplace=True)
test_csv.dropna(axis=0,inplace=True)

var=train_csv.corr()
var
# Identifying the correlation between the features :

correlation=train_csv.corr()
sns.heatmap(correlation)

plt.gray() 
plt.matshow(rand_img[0]) # change the values in the arary and can visualize the dataset
plt.show()

import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
import numpy as np
from sklearn.decomposition import TruncatedSVD
float_formatter = lambda x: "%.2f" % x
np.set_printoptions(formatter={'float_kind':float_formatter})
from sklearn.ensemble import RandomForestClassifier
# no need to paste in doc

import random
from matplotlib.image import imread
 # no need to paste in doc

data_dir='/content/drive/MyDrive/DataScience'
train_path='/content/drive/MyDrive/DataScience/Train.csv'

#test_path='../input/gtsrb-german-traffic-sign/'
test = pd.read_csv(data_dir+ '/Train.csv')
imgs = test["Path"].values

plt.figure(figsize=(25,25))

for i in range(1,26):
    plt.subplot(5,5,i)
    random_img_path = data_dir + '/' + random.choice(imgs)
    rand_img = imread(random_img_path)
    plt.imshow(rand_img)
    plt.grid(b=None)
    plt.xlabel(rand_img.shape[1], fontsize = 20)#width of image
    plt.ylabel(rand_img.shape[0], fontsize = 20)#height of image

plt.gray()

plt.matshow(rand_img[0]) # change the values in the arary and can visualize the dataset
plt.show()

image=rand_img[0]
image

plt.matshow(rand_img) # change the values in the arary and can visualize the dataset
plt.show()
image=rand_img[0]
image
image = image.reshape((6,16))
image
plt.matshow(image, cmap = 'gray')

U, s, V = np.linalg.svd(image)
S = np.zeros((image.shape[0], image.shape[1]))
S

S[:image.shape[0], :image.shape[0]] = np.diag(s) #diagonal matrix, by default it will create nxn matrix
S

n_component = 2
S = S[:, :n_component]
S

V = V[:n_component, :]
V

A = U.dot(S.dot(V))
print(A)

plt.matshow(A, cmap = 'gray')
U.dot(S) #

