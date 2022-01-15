#!/usr/bin/env python
# coding: utf-8

# <H1>SEP 787 - Final Project</H1>
# 
# <H3>Team:</H3>
# Hilary Luo 400355805 <br>
# Kevin Chong 400071824 <br>
# Wajahat Ali Khan 400346513 <br>
# 
# 
# 
# 

# <h2>Data Description</h2>
# 
# Contributer's description:
# >Seven different types of dry beans were used in this research, taking into account the features such as form, shape, type, and structure by the market situation. A computer vision system was developed to distinguish seven different registered varieties of dry beans with similar features in order to obtain uniform seed classification. For the classification model, images of 13,611 grains of 7 different registered dry beans were taken with a high-resolution camera. Bean images obtained by computer vision system were subjected to segmentation and feature extraction stages, and a total of 16 features; 12 dimensions and 4 shape forms, were obtained from the grains.
# 
# Data was retrieved from <link>https://archive.ics.uci.edu/ml/datasets/Dry+Bean+Dataset#</link>
# 
# <h3>Attributes</h3>
# <ol>
# <li>Area (A): The area of a bean zone and the number of pixels within its boundaries.</li> 
# <li>Perimeter (P): Bean circumference is defined as the length of its border.</li>
# <li>Major axis length (L): The distance between the ends of the longest line that can be drawn from a bean.</li>
# <li>Minor axis length (l): The longest line that can be drawn from the bean while standing perpendicular to the main axis.</li>
# <li>Aspect ratio (K): Defines the relationship between L and l.</li>
# <li>Eccentricity (Ec): Eccentricity of the ellipse having the same moments as the region.</li>
# <li>Convex area (C): Number of pixels in the smallest convex polygon that can contain the area of a bean seed.</li>
# <li>Equivalent diameter (Ed): The diameter of a circle having the same area as a bean seed area.</li>
# <li>Extent (Ex): The ratio of the pixels in the bounding box to the bean area.</li>
# <li>Solidity (S): Also known as convexity. The ratio of the pixels in the convex shell to those found in beans.</li>
# <li>Roundness (R): Calculated with the following formula: (4piA)/(P^2)</li>
# <li>Compactness (CO): Measures the roundness of an object: Ed/L</li>
# <li>ShapeFactor1 (SF1)</li>
# <li>ShapeFactor2 (SF2)</li>
# <li>ShapeFactor3 (SF3)</li>
# <li>ShapeFactor4 (SF4)</li>
# <li>Class (Seker, Barbunya, Bombay, Cali, Dermosan, Horoz and Sira)</li>
# </ol>
# 
# 

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.svm, sklearn.metrics
import time

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import KFold, train_test_split


# <h2>Data Loading</h2>
# Retrieving the data, selecting the two classes that we will be using and dividing it into sections for cross validation and testing. 

# In[3]:


FILE_NAME = 'Dry_Bean_Dataset.xlsx'

df_raw = pd.read_excel(FILE_NAME)

df_class_0 = df_raw['Class'] == 'SEKER'
df_class_1 = df_raw['Class'] == 'BARBUNYA'
df = df_raw[df_class_0 | df_class_1]
df = df.sample(frac=1).reset_index(drop=True) #shuffle the rows
df_class_mask = df['Class'] == "BARBUNYA" #Keeping the labels to check results later

df = df.drop(['Class'], axis=1) #Remove the class from the data table leaving only attributes

X_train, X_test, y_train, y_test = train_test_split(df.to_numpy(), df_class_mask.to_numpy(), test_size=0.25)

#Set up the cross validation segments
training_data = []
training_class = []
validation_data = []
validation_class = []

kf = KFold(n_splits = 5)

for train_index, validate_index in kf.split(X_train):
    training_data.append(X_train[train_index])
    training_class.append(y_train[train_index])
    validation_data.append(X_train[validate_index])
    validation_class.append(y_train[validate_index])


# <h2>Support Vector Machine Approach</h2>
# <h3>Choosing the hyperparameter C and Kernel</h3>
# Cross validation is used to evaluate different values of C and to determine the optimal value. This was repeated for both the RBF and linear kernels.
# <h4>Radial Basis Function SVM</h4>

# In[51]:


performance = []
for j in range(1,200,10):
    svm = sklearn.svm.SVC(C=(j*j), kernel='rbf')
    avg_score = 0
    avg_time = 0
    avg_train_time = 0

    for i in range(5):
        #Training
        start = time.perf_counter()
        for k in range(10):
            svm.fit(training_data[i],training_class[i])
        end = time.perf_counter()
        avg_train_time = avg_train_time + (end - start)

        #Validation
        start = time.perf_counter()
        for k in range(10):
            svm.predict(validation_data[i])
        end = time.perf_counter()
        avg_time = avg_time + (end - start)
        avg_score = avg_score + svm.score(validation_data[i],validation_class[i])

    avg_score = avg_score/5
    avg_time = avg_time/50
    avg_train_time = avg_train_time/50
    performance.append([(j*j),avg_score,avg_train_time,avg_time])

perf_df = pd.DataFrame(performance)
plt.plot(perf_df.iloc[:,0],perf_df.iloc[:,1])
plt.title('RBF SVM Cross Validation Score')
plt.xlabel('Hyperparameter (C)')
plt.ylabel('Classification Score')
plt.show()
plt.plot(perf_df.iloc[:,0],perf_df.iloc[:,2], label = 'Training Time')
plt.plot(perf_df.iloc[:,0],perf_df.iloc[:,3], label = 'Testing Time')
plt.title('RBF SVM Cross Validation Timing')
plt.xlabel('Hyperparameter (C)')
plt.ylabel('Time (sec)')
plt.legend()
plt.show()


# <h4>Linear SVM</h4>

# In[50]:


performance = []
for j in range(1,200,20):
    svm = sklearn.svm.SVC(C=j/10, kernel='linear')
    avg_score = 0
    avg_time = 0
    avg_train_time = 0

    for i in range(5):
        #Training
        start = time.perf_counter()
        for k in range(3):
            svm.fit(training_data[i],training_class[i])
        end = time.perf_counter()
        avg_train_time = avg_train_time + (end - start)

        #Validation
        start = time.perf_counter()
        for k in range(3):
            svm.predict(validation_data[i])
        end = time.perf_counter()
        avg_time = avg_time + (end - start)
        avg_score = avg_score + svm.score(validation_data[i],validation_class[i])

    avg_score = avg_score/5
    avg_time = avg_time/15
    avg_train_time = avg_train_time/15
    performance.append([j/10,avg_score,avg_train_time,avg_time])

perf_df = pd.DataFrame(performance)
plt.plot(perf_df.iloc[:,0],perf_df.iloc[:,1])
plt.title('Linear SVM Cross Validation Score')
plt.xlabel('Hyperparameter (C)')
plt.ylabel('Classification Score')
plt.show()
plt.plot(perf_df.iloc[:,0],perf_df.iloc[:,2], label = 'Training Time')
plt.plot(perf_df.iloc[:,0],perf_df.iloc[:,3], label = 'Testing Time')
plt.title('Linear SVM Cross Validation Timing')
plt.xlabel('Hyperparameter (C)')
plt.ylabel('Time (sec)')
plt.legend()
plt.show()
plt.plot(perf_df.iloc[:,0],perf_df.iloc[:,3], label = 'Testing Time')
plt.title('Linear SVM Cross Validation Testing Timing')
plt.xlabel('Hyperparameter (C)')
plt.ylabel('Time (sec)')
plt.show()


# <h3>Selected RBF Hyperparameter</h3>
# C value of 12500 was selected to optimize score without taking excessive time to train
# 
# 

# In[5]:


svm = sklearn.svm.SVC(C=12500, kernel='rbf')

#Training
start = time.perf_counter()
for k in range(10):
    svm.fit(X_train,y_train)
end = time.perf_counter()
svm_train_time = (end - start)/10

#Testing
start = time.perf_counter()
for k in range(10):
    svm_predict = svm.predict(X_test)
end = time.perf_counter()
svm_test_time = (end - start)/10

print('The training process took ',svm_train_time*1000,' ms.\nThe testing process took ', svm_test_time*1000,' ms.\n')

#ROC
sklearn.metrics.plot_roc_curve(svm,X_test,y_test)

#Confusion Matrix
print('Confusion Matrix:\n',sklearn.metrics.confusion_matrix(y_test,svm_predict),'\n')
plt.title('RBF SVM Receiver Operating Curve (ROC)')
plt.show()


# <h3>Selected Linear Hyperparameter</h3>
# C value of 6 was selected to optimize score without taking excessive time to train

# In[6]:


svm = sklearn.svm.SVC(C=6, kernel='linear')

#Training
start = time.perf_counter()
for k in range(10):
    svm.fit(X_train,y_train)
end = time.perf_counter()
svm_train_time = (end - start)/10

#Testing
start = time.perf_counter()
for k in range(10):
    svm_predict = svm.predict(X_test)
end = time.perf_counter()
svm_test_time = (end - start)/10

print('The training process took ',svm_train_time*1000,' ms.\nThe testing process took ', svm_test_time*1000,' ms.\n')

#ROC
sklearn.metrics.plot_roc_curve(svm,X_test,y_test)

#Confusion Matrix
print('Confusion Matrix:\n',sklearn.metrics.confusion_matrix(y_test,svm_predict),'\n')
plt.title('Linear SVM Receiver Operating Curve (ROC)')
plt.show()


# <h2>K Nearest Neighbor Approach</h2>

# In[6]:


from sklearn.neighbors import KNeighborsClassifier

results = []

nieghbors = [1, 3, 6, 9, 12, 15, 50, 100, 1000]
for n in nieghbors:
    print("For a K value of ", n)
    knn = KNeighborsClassifier(n_neighbors = n)
    results = []
    for i in range(5):
        print('\nKNN Cross Validation Set ',i+1)
        get_ipython().run_line_magic('timeit', '-n10 -r5 knn.fit(training_data[i],training_class[i])')
        get_ipython().run_line_magic('timeit', '-n10 -r5 results = knn.predict(validation_data[i])')

        sklearn.metrics.plot_roc_curve(knn,validation_data[i],validation_class[i])
        plt.show()


# In[7]:


knn_best = KNeighborsClassifier(n_neighbors = 1)
knn_best.fit(X_train, y_train)

predict = knn_best.predict(X_test)

print(sklearn.metrics.confusion_matrix(y_test, predict))


# <h2>Adaboost Approach</h2>

# In[4]:


from sklearn.ensemble import AdaBoostClassifier
results = []
average_score = []
average_time = []
average_test_time = []
n_estimators = [1, 3, 9, 15, 20, 25, 30, 40, 50]
for n in n_estimators:
    print('\n Number of decision trees',n)
    abc=AdaBoostClassifier(n_estimators=n)
    results=[]
    train_time=0
    test_time=0
    avg_score=0

    for i in range(5):
        #print('\nAdaboost Cross Validation Set ',i+1)
        
        start = time.perf_counter()
        abc.fit(training_data[i],training_class[i])
        end = time.perf_counter()
        train_time = train_time + ((end - start)/5)
        
        start = time.perf_counter()
        results = abc.predict(validation_data[i])
        end = time.perf_counter()
        test_time = test_time + ((end - start)/5)
        
        score=abc.score(validation_data[i],validation_class[i])
        avg_score=avg_score+(score/5)
        
        
    average_score.append(avg_score)
    print('The average score for' ,n, 'decision trees is ',avg_score)
    
    average_time.append(train_time)
    print('The average training time for' ,n, 'decision trees is ',train_time)
    
    average_test_time.append(test_time)
    print('The average testing time for' ,n, 'decision trees is ',test_time)    


# In[8]:


print(average_score)


# In[7]:


plt.plot(n_estimators,average_score)
plt.xlabel('Number of trees (n)')
plt.ylabel('Classification Score')
plt.title('Adaboost cross validation scores')
plt.show()
plt.plot(n_estimators,average_time)
plt.xlabel('Number of trees (n)')
plt.ylabel('Average training time')
plt.title('Adaboost cross validation training time')
plt.show()
plt.plot(n_estimators,average_test_time)
plt.xlabel('Number of trees')
plt.ylabel('Average testing time')
plt.title('Adaboost cross validation testing time')
plt.show()


# In[ ]:


abc_best=AdaBoostClassifier(n_estimators=40)
get_ipython().run_line_magic('timeit', '-n10 -r5 abc_best.fit(X_train,y_train)')
get_ipython().run_line_magic('timeit', '-n10 -r5 predict = abc_best.predict(X_test)')
sklearn.metrics.plot_roc_curve(abc,validation_data[i],validation_class[i])
plt.show()
    
print(sklearn.metrics.confusion_matrix(y_test, predict))


# In[ ]:




