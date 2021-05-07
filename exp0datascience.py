# -*- coding: utf-8 -*-
"""Exp0DataScience.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1zIv5bHm9dC4UUzcU9bVA_YQr0ulgxRyV
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

test_path = '/content/drive/MyDrive/DataScience/Test'
test_img = sorted(os.listdir(test_path))

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout, Conv2D, MaxPool2D

import numpy as np
import pandas as pd

images = np.load('Training_set.npy')
label_id = np.load('Label_Id.npy')

model = Sequential()

#1st layer
model.add(Conv2D(filters = 64, kernel_size = (3,3), input_shape = x_train.shape[1:], activation = 'relu', padding = 'same'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.5))

#2nd layer
model.add(Conv2D(filters = 64, kernel_size = (3,3), activation = 'relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.5))

#3rd layer
model.add(Conv2D(filters = 64, kernel_size = (3,3), activation = 'relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.5))

model.add(Flatten())

#Dense layer
model.add(Dense(128, activation = 'relu'))
model.add(Dropout(0.5))

#Output layer
model.add(Dense(43, activation = 'softmax'))
model.compile(loss = 'sparse_categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])





from PIL import Image

def scaling(test_images, test_path):
    images = []

    image_path = test_images
    
    for x in image_path:
        img = Image.open(test_path + '/' + x)
        img = img.resize((50,50))
        img = np.array(img)
        images.append(img)

    #Converting images into numpy array
    images = np.array(images)
    #The pixel value of each image ranges between 0 and 255
    #Dividing each image by 255 will scale the values between 0 and 1. This is also known as normalization.
    images = images/255

    return images

test_images = scaling(test_img,test_path)

test = pd.read_csv('/content/drive/MyDrive/DataScience/Test.csv')

y_test = test['ClassId'].values

y_test

len(y_test)

print(test_csv)

y_pred = model.predict_classes(test_images)

print(y_pred)

predict=np.argmax(Model.predict(test),axis=-1)

predict

len(predict)

label = test_csv['ClassId'].values

len(labels)

print('Test Data Accuracy',accuracy_score(labels[:-8906],prediction)*100)



meta_csv['ColorId'].value_counts() #to print the colorId value and it's respective count

min(meta_csv['ColorId'].value_counts()) #to print the min no of ColorIds present

max(meta_csv['ColorId'].value_counts()) #to print the max no of ColorIds present

import pandas as pd
meta_csv=pd.read_csv('/content/drive/MyDrive/DataScience/Meta.csv') 
meta_csv.head(10)    
#to print the is used to get the first 10 rows

len(meta_csv) # is used to print  the total number of rows / length of dataset

meta_csv=pd.read_csv('/content/drive/MyDrive/DataScience/Meta.csv') 
print (meta_csv['ShapeId'].value_counts()) # to print the values present in ShapeId and their count

meta_csv=pd.DataFrame('/content/drive/MyDrive/DataScience/Meta.csv',columns =['Path', 'ClassId', 'ShapeId','ColorId','SignId'],index =['0', '1', '2', '3','4']) 
for index in range(meta_csv.shape[1]): 
     print('Column Number : ', index) 
for column in meta_csv: 
    print('Colunm Name : ', column)                   
  #to print the column name and number using control strusture

datascience=pd.read_csv('/content/drive/MyDrive/DataScience')

train_path=pd.read_csv('/content/drive/MyDrive/DataScience/Train.csv') 
test = pd.read_csv('/content/drive/MyDrive/DataScience/Train.csv')
imgs = test["Path"].values
plt.figure(figsize=(25,25))

for i in range(1,26):
  plt.subplot(5,5,i)
  random_img_path = datascience + '/' + random.choice(imgs)
  rand_img = imread(random_img_path)
  plt.imshow(rand_img)
  plt.grid(b=None)
  plt.xlabel(rand_img.shape[1], fontsize = 20)#width of image
  plt.ylabel(rand_img.shape[0], fontsize = 20)#height of image

"""Experiment 1 ends



"""

os.getcwd() # to returns that contains the absolute path of the current working directory

os.listdir() # list files in the folder

.path.exists('/conostent/drive/MyDrive/DataScience/data.csv')  # check if a file exists

os.path.exists('/content/drive/MyDrive/DataScience/Meta.csv') # check if a file exists

test_csv=pd.read_csv('/content/drive/MyDrive/DataScience/Test.csv')  
test_csv.head()
test_csv.head(7)
# check first seven rows of data

test_csv.tail()
# display last n rows of data by default

test_csv.tail(10)
# display last n rows of data by default

test_csv.shape # display shape of the array

data1 = list(test_csv["Height"])
print(data1)
# Height is treated as list and is printed

tup=tuple(test_csv["Height"])
print(tup)
# Height is treated as tuple and is printed

se=set(test_csv["Height"])
print(se)
# Height is treated as set and is printed

dic = dict(test_csv)
print(dic)
# Height is treated as dictionary and is printed

"""Expertment 2 ends

"""

meta2_csv=meta_csv.copy();

meta_csv.dropna(axis=0,inplace=True)	#Handling missing data :

train_csv.dropna(axis=0,inplace=True)

test_csv.dropna(axis=0,inplace=True)

plt.scatter(meta_csv['ShapeId'],meta_csv['ClassId'],c='red')
plt.title('Scatter plot')
plt.xlabel('ShapeId')
plt.ylabel('ClassId')
plt.show()
# Scatter Plot :

plt.hist(train_csv['ClassId'])
plt.hist(train_csv['ClassId'],color='green',edgecolor='white',bins=6)
plt.title('Histogram')
plt.xlabel('Class Label')
plt.ylabel('Frequency')
plt.show()
#Histogram

ShapeId=(0,1,2,3,4)
y=np.arange(len(ShapeId))
index=3
ClassId=(20,34,40)
plt.bar(index,y,color=['red','blue','cyan'])
plt.xticks(y,ShapeId)
plt.ylabel('ClassId')
plt.show()

ShapeId=(0,1,2,3,4)
y=np.arange(len(ShapeId))
ClassId=('20','4','11','0','8','20','21')
index=3
plt.bar(index,y,color=['red','blue'])
plt.title('Bar Plot')
plt.xlabel('ClassId')
plt.ylabel('Frequency')

plt.bar(index,y,color=['red','blue'])
plt.title('Bar Plot')
plt.xlabel('ClassId')
plt.ylabel('Frequency')
plt.xticks(index,ClassId,rotation=90)
plt.show()
#bar graph

numerical_data=train_csv.select_dtypes(exclude=[object])
print(numerical_data.shape)
corr_matrix=numerical_data.corr()
 #Correlation between numerical variables

sns.set(style="darkgrid")  #SNS

sns.regplot(x=meta_csv['ShapeId'],y=meta_csv['ClassId'])   #SNS

sns.regplot(x=meta_csv['ShapeId'],y=meta_csv['ClassId'],fit_reg=False)   #SNS

sns.regplot(x=meta_csv['ShapeId'],y=meta_csv['ClassId'],fit_reg=False,marker="*")  #SNS

sns.lmplot(x='ShapeId',y='ClassId',data=meta_csv,fit_reg=False,hue='ColorId',legend=True,palette='Set1')  #SNS

sns.distplot(meta_csv['ShapeId'])
sns.distplot(meta_csv['ShapeId'],kde=False)
sns.distplot(meta_csv['ShapeId'],kde=False,bins=5)

#SNS-Histogram

sns.countplot(x="ShapeId",data=meta_csv)
sns.countplot(x="ShapeId",data=meta_csv,hue="ColorId")
#SNS- Grouped bar plot

sns.boxplot(y=meta_csv["ShapeId"])
sns.boxplot(x=meta_csv["ShapeId"],y=meta_csv["ClassId"])
sns.pairplot(meta_csv,kind="scatter",hue="ClassId")
plt.show()
#SNS - box whiskers and pairwise plot

"""Experiment 3 ends"""

train_csv.describe()

train_csv.describe(include="O")

var=train_csv.corr()
var

correlation=train_csv.corr()
sns.heatmap(correlation)

Add the program from mail

fig=plt.figure(figsize=(8,8))
ax=fig.add_subplot(1,1,1)
ax.set_xlabel('Principal Component 1',fontsize=15)
ax.set_ylabel('Principal Component 2',fontsize=15)
ax.set_title('2 Component PCA',fontsize=20)
ClassIds=[20,11,14]
colors=['r','g','b']
for ClassId,color in zip(ClassIds,colors):
        indicesToKeep=finalDf['ClassId']==ClassId
        ax.scatter(finalDf.loc[indicesToKeep,'principal component 1'],finalDf.loc[indicesToKeep,'principal component 2'],c=color,s=50)
        ax.legend(ClassIds)
        ax.grid() #Final PCA output

train_csv.dropna()
print(train_csv.shape)
print(train_csv.columns)

#Shape and attributes in train dataset

train_csv['ClassId'].value_counts()
sns.countplot(x='ClassId',data=train_csv)
train_csv.groupby('ClassId').mean()
train_csv['ClassId']=test_csv['ClassId'].astype('category') #used to display the components in ClassId

train_csv['ClassId'].value_counts()
sns.countplot(x='ClassId',data=train_csv)
train_csv.groupby('ClassId').mean() # to groupby ClassId and display counts of observations in each categorical bin using bars

import pandas as pd
import numpy as np
import sklearn
from scipy import stats
import matplotlib.pyplot as plt
import os
import seaborn as sns
# no need to paste in doc

sns.countplot(x='ClassId',data=test_csv)
test_csv.groupby('ClassId').mean()   # to groupby ClassId and display counts of observations in each categorical bin using bars

test_csv['ClassId']=test_csv['ClassId'].astype('category')
test_csv['ClassId']=test_csv['ClassId'].cat.codes
test_csv

test_csv['Path']=test_csv['Path'].astype('category')
test_csv['Path']=test_csv['Path'].cat.codes
test_csv

from sklearn.svm import LinearSVC, SVC
from numpy import *

svm = LinearSVC()
# create the RFE model for the svm classifier and select attributes

from sklearn.datasets import make_friedman1
from sklearn.feature_selection import RFE
from sklearn.svm import SVR
import os,sys
from scipy import stats
import numpy as np
# no need to paste in doc

type(test_csv) # no need to paste in doc

rfe = RFE(svm, 6)
rfe = rfe.fit(test_csv, test_csv.ClassId)
# summaries for the selection of attributes

  

print(rfe.support_)
print(rfe.ranking_)
rank=rfe.ranking_
colms=train_csv.columns.values

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

plt.matshow(rand_img) # change the values in the arary and can visualize the dataset
plt.show()
image=rand_img[0]
image
image = image.reshape((6,17))
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

"""Experiment 4 ends

### **EXPT 5 starts**
"""

import numpy as np
from matplotlib import pyplot as plt
from sklearn.neighbors import KernelDensity

meta_csv=pd.read_csv('/content/drive/MyDrive/DataScience/Meta.csv')   #Read the file
test_csv=pd.read_csv('/content/drive/MyDrive/DataScience/Test.csv')    #Read the file
train_csv=pd.read_csv('/content/drive/MyDrive/DataScience/Train.csv')   #Read the file

def kde_sklearn(x, x_grid, bandwidth=0.2, **kwargs):
    """Kernel Density Estimation with Scikit-learn"""
    kde_skl = KernelDensity(bandwidth=bandwidth, **kwargs)  
    kde_skl.fit(x[:, np.newaxis])
    log_pdf = kde_skl.score_samples(x_grid[:, np.newaxis])
    return np.exp(log_pdf)

df=train_csv['Width'].copy()
plt.hist(df, bins=100,alpha=0.9 )
x_grid = np.linspace(-1, 1, 100)
pdf_engagement = kde_sklearn(df, x_grid, bandwidth=0.007)
#

plt.hist(df, bins=100,alpha=0.9)
x_grid = np.linspace(15, 200, 5)
pdf_engagement = kde_sklearn(df, x_grid, bandwidth=0.007)
plt.plot(x_grid, pdf_engagement,alpha=0.9, lw=5, color='r')
plt.xlabel("Width")
plt.ylabel("ClassId")
plt.show()
# more than 2500 samples have width of image less than 50

x = train_csv['Height'].copy()
mean = np.mean(x)
std = np.std(x)
print("""Height mean: %.5f
Height std: %.5f
Height size: %i
"""%(mean, std, len(x)))
# mean and std of height attribute is calculted

sample_size = 400
n_trials = 50000
# draw one million samples, each of size 300
samples = np.array([np.random.choice(df, sample_size)
for _ in range(n_trials)])
# calculate sample mean for each sample
means = samples.mean(axis=1)
# mean of sampling distribution
sample_mean = np.mean(means)
# empirical standard error
sample_std = np.std(means)
analytical_std = std / np.sqrt(sample_size)
print("""
sampling distribution mean: %.5f
sampling distribution std: %.5f
analytical std: %.5f
"""%(sample_mean, sample_std, analytical_std))

#https://towardsdatascience.com/understanding-confidence-interval-d7b5aa68e3b to write story
#In the real world, we would consider this dataset as a sample of size 8702. But in this simulation, we treat it as if we’re the entire population, and draw one million samples of size 300 from it (a process known as Bootstrap sampling). Because we have the full population here, we can easily see that the mean of the sampling distribution is unbiased estimator of population mean (0.07727). Furthermore, it’s easy to confirm that the empirically computed standard error is identical to the analytical one (0.0062).
#"Is the paragraph to be modified" along with the following paragraphs can be included

# make 95% confidence interval
z = 1.96

se = samples.std(axis=1) / np.sqrt(sample_size)
ups = means + z * se
los = means - z * se

success = np.mean((mean >= los) & (mean <= ups))
fpr = np.mean((mean < los) | (mean > ups))

print("False positive rate: %.3f"%fpr)

# sampling distribution
from scipy.stats import t
z = 1.96
plt.hist(means, bins=50, alpha=0.9)
plt.axvline(sample_mean - 1.96 * sample_std, color='r')
plt.axvline(sample_mean + 1.96 * sample_std, color='r')
plt.xlabel('Width')
plt.ylabel('ClassId')
plt.show()
print("lower tail: %.2f%%"%(100 * sum(means < sample_mean - 1.96 * sample_std) /
len(means)))
print("upper tail: %.2f%%"%(100 * sum(means > sample_mean + 1.96 * sample_std) /
len(means)))
#

import pylab 
import scipy.stats as stats

stats.probplot(means, dist="norm", plot=pylab)
pylab.show()
#QQ plot confirms tendency toward normal distribution. It is a scatterplot created by plotting two sets of quantiles against one another.

n_points = 4000

plt.figure(figsize=(14, 6))
plt.scatter(list(range(len(ups[:n_points]))), ups[:n_points], alpha=0.3)
plt.scatter(list(range(len(los[:n_points]))), los[:n_points], alpha=0.3)
plt.axhline(y=0.07727)
plt.xlabel("sample")
plt.ylabel("sample_mean")

#approximately 5% of the confidence intervals fail to capture the population mean. In the graph below, this happens when the blue dots (upper bound) cross below the population mean, or when orange dots (lower bound) cross above the population mean.

"""Testing

"""

#(Anderson testing)
from scipy.stats import anderson
data1 = train_csv['ClassId']
result = anderson(data1)
print('stat=%.3f' % (result.statistic))
for i in range(len(result.critical_values)):
  sl, cv = result.significance_level[i], result.critical_values[i]
  if result.statistic < cv:
    print('Probably Gaussian at the %.1f%% level' % (sl))
  else:
    print('Probably not Gaussian at the %.1f%% level' % (sl))
    # Anderson-Darling Test - Tests whether a data sample has a Gaussian distribution.

#Shapiro testing
from scipy.stats import shapiro

stat, p = shapiro(df)
print('stat=%.3f, p=%.3f' % (stat, p))
if p > 0.05:
    print('Probably Gaussian')
else:
    print('Probably not Gaussian')
#    Shapiro-Wilk Test: Tests whether a data sample has a Gaussian distribution.

#D’Agostino’s K^2 Test - Tests whether a data sample has a Gaussian distribution.

from scipy.stats import normaltest

stat, p = normaltest(x)
print('stat=%.3f, p=%.3f' % (stat, p))
if p > 0.05:
    print('Probably Gaussian')
else:
    print('Probably not Gaussian')

#  Pearson's Correlation test
from scipy.stats import pearsonr
data1 = train_csv['Width']
data2 = train_csv['ClassId']

stat, p = pearsonr(data1, data2)
print('stat=%.3f, p=%.3f' % (stat, p))
if p > 0.05:
	print('Probably independent')
else:
	print('Probably dependent')
 # Pearson’s Correlation Coefficient-Tests whether two samples have a linear relationship

#  Spearman's Rank Correlation Test
from scipy.stats import spearmanr
data1 = train_csv['Width']
data2 = train_csv['ClassId']

stat, p = spearmanr(data1, data2)
print('stat=%.3f, p=%.3f' % (stat, p))
if p > 0.05:
	print('Probably independent')
else:
	print('Probably dependent')
 # Spearman’s Rank Correlation - Tests whether two samples have a monotonic relationship.

# Kendall’s Rank Correlation - Tests whether two samples have a monotonic relationship.
from scipy.stats import kendalltau
data1 = train_csv['Width']
data2 = train_csv['ClassId']

stat, p = kendalltau(data1, data2)
print('stat=%.3f, p=%.3f' % (stat, p))
if p > 0.05:
	print('Probably independent')
else:
	print('Probably dependent')

# Augmented Dickey-Fuller Unit Root Test 

from statsmodels.tsa.stattools import adfuller
data1 = train_csv['Width']
stat, p, lags, obs, crit, t = adfuller(data1)
print('stat=%.3f, p=%.3f' % (stat, p))
if p > 0.05:
	print('Probably not Stationary')
else:
	print('Probably Stationary')

# Kwiatkowski-Phillips-Schmidt-Shin - Tests whether a time series is trend stationary or not.


from statsmodels.tsa.stattools import kpss
data = train_csv['Height']
stat, p, lags, crit = kpss(data)
print('stat=%.3f, p=%.3f' % (stat, p))
if p > 0.05:
	print('Probably not Stationary')
else:
	print('Probably Stationary')

# 4. Parametric Statistical Hypothesis Tests
# Student’s t-test - Tests whether the means of two independent samples are significantly different.
from scipy.stats import ttest_ind
data1 = train_csv['Width']
data2 = train_csv['ClassId']
stat, p = ttest_ind(data1, data2)
print('stat=%.3f, p=%.3f' % (stat, p))
if p > 0.05:
	print('Probably the same distribution')
else:
	print('Probably different distributions')

"""# Paired Student’s t-test - Tests whether the means of two paired samples are significantly different."""
from scipy.stats import ttest_rel
data1 = train_csv['Width']
data2 = train_csv['ClassId']
stat, p = ttest_rel(data1, data2)
print('stat=%.3f, p=%.3f' % (stat, p))
if p > 0.05:
	print('Probably the same distribution')
else:
	print('Probably different distributions')

# One Way - Analysis of Variance Test (ANOVA)
from scipy.stats import f_oneway
data1 = train_csv['Width']
data2 = train_csv['ClassId']
data3 = train_csv['Height']
stat, p = f_oneway(data1, data2, data3)
print('stat=%.3f, p=%.3f' % (stat, p))
if p > 0.05:
	print('Probably the same distribution')
else:
	print('Probably different distributions')

# 5. Nonparametric Statistical Hypothesis Tests
# Mann-Whitney U Test - Tests whether the distributions of two independent samples are equal or not.

from scipy.stats import mannwhitneyu
data1 = train_csv['Width']
data2 = train_csv['Height']
stat, p = mannwhitneyu(data1, data2)
print('stat=%.3f, p=%.3f' % (stat, p))
if p > 0.05:
	print('Probably the same distribution')
else:
	print('Probably different distributions')

# Wilcoxon Signed-Rank Test - Tests whether the distributions of two paired samples are equal or not.
from scipy.stats import wilcoxon
data1 = train_csv['Width']
data2 = train_csv['ClassId']
stat, p = wilcoxon(data1, data2)
print('stat=%.3f, p=%.3f' % (stat, p))
if p > 0.05:
	print('Probably the same distribution')
else:
	print('Probably different distributions')

# Kruskal-Wallis H Test - Tests whether the distributions of two or more independent samples are equal or not.
from scipy.stats import kruskal
data1 = train_csv['Width']
data2 = train_csv['ClassId']
stat, p = kruskal(data1, data2)
print('stat=%.3f, p=%.3f' % (stat, p))
if p > 0.05:
	print('Probably the same distribution')
else:
	print('Probably different distributions')

# Friedman Test - Tests whether the distributions of two or more paired samples are equal or not.
from scipy.stats import friedmanchisquare
data1 = train_csv['Width']
data2 = train_csv['ClassId']
data3 = train_csv['Height']
stat, p = friedmanchisquare(data1, data2, data3)
print('stat=%.3f, p=%.3f' % (stat, p))
if p > 0.05:
	print('Probably the same distribution')
else:
	print('Probably different distributions')

import matplotlib as plt
from pandas.plotting import scatter_matrix
scatter_matrix(data[['Width', 'Height', 'Roi.Y2']])

import pandas as pd
data =train_csv=pd.read_csv('/content/drive/MyDrive/DataScience/Train.csv')  
data

import numpy as np
t = np.linspace(-6, 6, 20)
sin_t = np.sin(t)
cos_t = np.cos(t)
pd.DataFrame({'t': t, 'sin': sin_t, 'cos': cos_t})
data.shape

data.shape
data.columns
print(data['ClassId'])

data.describe()

groupby_class = data.groupby('ClassId')
for classid, value in groupby_class['Height']:
  print(classid, value.mean())

groupby_class.mean()

import matplotlib as plt
from pandas.plotting import scatter_matrix
scatter_matrix(data[['Width', 'Height', 'Roi.Y2']])

scatter_matrix(data[['Roi.X1', 'Roi.X2', 'Roi.Y1']])

"""# **Extras**

"""

#1.	Probability:
import warnings
warnings.filterwarnings('ignore')
test_csv.shape
total_val = test_csv.shape[0]
print("Total Number of values :", total_val)
test_csv['Height'].value_counts()
age = (test_csv['Height'] >=50).sum()
print("No of traffic signs with Height greater than 50:",age)
probability22 = (age/total_val)*100
print('Probability of displaying a traffic signs with Height greater than 50 : {0:.2f}'.format(probability22 )+'%')

cond_prob_22 = (age/total_val) * ((age - 1)/(total_val - 1)) 
print("The Probability of displaying a traffic signs with Height greater than 50  and againdisplaying a traffic signs with Height greater than 50  {0:.3f}".
      format(cond_prob_22*100))

import scipy.stats as stats
import math
# lets seed the random values
np.random.seed(10)
# lets take a sample size
sample_size = 1000
sample = np.random.choice(a= test_csv['Width'],
                          size = sample_size)
sample_mean = sample.mean()
# Get the z-critical value*
z_critical = stats.norm.ppf(q = 0.95)  
 # Check the z-critical value  
print("z-critical value: ",z_critical)                                
# Get the population standard deviation
pop_stdev = test_csv['Width'].std()  
# checking the margin of error
margin_of_error = z_critical * (pop_stdev/math.sqrt(sample_size)) 

# defining our confidence interval
confidence_interval = (sample_mean - margin_of_error,
                       sample_mean + margin_of_error)  

# lets print the results
print("Confidence interval:",end=" ")
print(confidence_interval)
print("True mean: {}".format(test_csv['Width'].mean()))
np.random.seed(12)
sample_size = 500
intervals = []
sample_means = []
for sample in range(25):
    sample = np.random.choice(a= test_csv['Width'], size = sample_size)
    sample_mean = sample.mean()
    sample_means.append(sample_mean)
     # Get the z-critical value* 
    z_critical = stats.norm.ppf(q = 0.97)         
    # Get the population standard deviation
    pop_stdev = test_csv['Width'].std()  
    stats.norm.ppf(q = 0.025)
    margin_of_error = z_critical * (pop_stdev/math.sqrt(sample_size))
    confidence_interval = (sample_mean - margin_of_error,
                           sample_mean + margin_of_error)  
    intervals.append(confidence_interval)
plt.figure(figsize=(13, 9))

plt.errorbar(x=np.arange(0.1, 25, 1), 
             y=sample_means, 
             yerr=[(top-bot)/2 for top,bot in intervals],
             fmt='o')

plt.hlines(xmin=0, xmax=25,
           y=test_csv['Width'].mean(), 
           linewidth=2.0,
           color="red")
plt.title('Confidence Intervals for 25 Trials', fontsize = 20)
plt.show()

#p value 
from statsmodels.stats.weightstats import ztest

z_statistic, p_value = ztest(x1 = test_csv[test_csv['Width'] >= 50]['ClassId'],
                             value = test_csv['ClassId'].mean())

# lets print the Results
print('Z-statistic is :{}'.format(z_statistic))
print('P-value is :{:.5f}'.format(p_value))

# If the P value if less than 0.05, then we can reject our null hypothesis against the alternate hypothesis.

#After this we can add "testing part"

"""### **EX 9 Starts**"""

import datetime
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense, Conv2D, MaxPool2D

import tensorflow as tf
print("TF version: ", tf.__version__)
train_df = pd.read_csv('/content/drive/MyDrive/DataScience/Train.csv')
train_df.describe()

train_df = train_df.drop(['Width', 'Height', 'Roi.X1', 'Roi.Y1', 'Roi.X2', 'Roi.Y2'], axis = 1)
train_df.head()

train_df['ClassId'].value_counts().plot.bar(figsize=(20, 10))
train_df['ClassId'].value_counts().median()

filenames = ['/content/drive/MyDrive/DataScience/' + fname for fname in train_df['Path']]
filenames[:10]

labels = train_df['ClassId'].to_numpy()
labels

IMG_SIZE = 32

def process_image(image_path):
    """
    Takes an image file path and turns the image into a Tensor.
    """
    # Read in an image file
    image = tf.io.read_file(image_path)
    # Turn the jpeg image into numerical Tensor with 3 colour channels (Red, Green, Blue)
    image = tf.image.decode_png(image, channels=3)
    # Convert the colour channel values from 0-255 to 0-1 values
    image = tf.image.convert_image_dtype(image, tf.float32)
    # Resize the image to our desired value (32, 32)
    image = tf.image.resize(image, size=[IMG_SIZE, IMG_SIZE])
    return image

def get_image_label (image_path, label):
    """
    Takes an image file path name and the assosciated label,
    processes the image and reutrns a typle of (image, label).
    """
    image = process_image(image_path)
    return image, label
BATCH_SIZE = 64

# Create a function to turn data into batches
def create_data_batches (X, y=None, batch_size=BATCH_SIZE, valid_data=False, test_data=False):
    """
    Creates batches of data out of image (X) and label (y) pairs.
    Shuffles the data if it's training data but doesn't shuffle if it's validation dat
    a.
    Also accepts test data as input (no labels).
    """
    # If the data is a test dataset, we probably don't have have labels
    if test_data:
        print("Creating test data batches...")
        data = tf.data.Dataset.from_tensor_slices((tf.constant(X)))
        data_batch = data.map(process_image).batch(BATCH_SIZE)
    # If the data is a valid dataset, we don't need to shuffle it
    elif valid_data:
        print("Creating validation dataset batches...")
        data = tf.data.Dataset.from_tensor_slices((tf.constant(X), tf.constant(y)))
        # Create (image, label) tuples (this also turns the iamge path into a preprocessed image)
        data_batch = data.map(get_image_label).batch(BATCH_SIZE)
    else:
        print("Creating training dataset batches...")
        # Turn filepaths and labels into Tensors
        data = tf.data.Dataset.from_tensor_slices((tf.constant(X), tf.constant(y)))
        # Shuffling pathnames and labels before mapping image processor function is faster than shuffling images
        data = data.shuffle(buffer_size=len(X))
        # Create (image, label) tuples (this also turns the iamge path into a preprocessed image) and turning into batches
        data_batch = data.map(get_image_label).batch(BATCH_SIZE)
    return data_batch

from sklearn.model_selection import train_test_split

# Creating training and validation batches
train_data = create_data_batches(X_train, y_train)
val_data = create_data_batches(X_val, y_val, valid_data=True)
# Check out the different attributes of our data batches
train_data.element_spec, val_data.element_spec

def show_25_images (images, labels):
    """
    Displays a plot of 25 images and their labels from a data batch.
    """
    plt.figure(figsize=(10,10))
    for i in range(25):
        ax = plt.subplot(5, 5, i+1)
        plt.imshow(images[i])
        plt.title(unique_signs[labels[i].argmax()])
        plt.axis("off")
train_images, train_labels = next(train_data.as_numpy_iterator())
show_25_images(train_images, train_labels)

from PIL import Image
def scaling(test_images, test_path):
    images = []
    image_path = test_image  
    for x in image_path:
        img = Image.open(test_path + '/' + x)
        img = img.resize((32,32))
        img = np.array(img)
        images.append(img)
    #Converting images into numpy array
    images = np.array(images)
    #The pixel value of each image ranges between 0 and 255
    #Dividing each image by 255 will scale the values between 0 and 1. This is also known as normalization.
    images = images/255
    return images
test_images = scaling(test_images,test_path)

from sklearn import metrics
import  seaborn as sns
df_cm = pd.DataFrame(cf, index =  classes, columns= classes)
plt.figure(figsize =(20,20))
sns.heatmap(df_cm, annot=True)
from sklearn.metrics import confusion_matrix
cf = confusion_matrix(labels[:-8906], prediction)

plt.figure(figsize=(25,25))
start_index = 0
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    prediction = y_pred[start_index + i]
    actual = labels[start_index + i]
    
    col = 'g'
    if prediction != actual:
       col = 'r'     
    plt.xlabel('Actual = {} || Pred = {}'.format(actual,prediction), color=col)
    plt.imshow(test_images[start_index + i])
plt.show()
