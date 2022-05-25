#First import the necessary libraries to build the model and preprocess the data

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#read the dataset from local storage using pandas 
data=pd.read_csv('breastcancer.csv')
data.head()
data.shape
data.info()   # All info regarding the dataset is available in this command
data.columns  # Lists out the columns name
data.isnull().sum() # Counts the number of null values presents in the dataset
data=data.drop('Unnamed: 32',axis=1)   # To eliminate the unnamed column it contains null values
data.columns  #listing other columns
data['diagnosis'].unique()     # it finds the unique values 
data['diagnosis'].value_counts()     # it counts unique values in diagnosis column
a=list(data.columns)
print(a)      # These columns are later stored as list
data.describe()   
#Plot the data in graph to get better understanding
sns.countplot(data['diagnosis'],data=None)
#Heatmap of Correlation
corr = data.corr()
plt.figure (figsize= (10,10))
sns.heatmap (corr);

data['diagnosis']=data['diagnosis'].map({'M':1,'B':0})   # To convert categorical values into numeric values
data['diagnosis'].head()

#Splitting the data into the Training and Testing set
x = data.drop ('diagnosis',axis=1)
y = data ['diagnosis']
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split (x,y,test_size=0.3)

x_train.shape()
x_test.shape()

#Feature Scaling of data
from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
x_train = ss.fit_transform (x_train)
x_test = ss.fit_transform (x_test)

#Importing Logistic Regression from Scikit learn library
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
#Loading the training data in the model
lr.fit (x_train, y_train)

#Predicting output with the test data
y_pred = lr.predict (x_test)
y_pred

#Accuracy Score of Logistic Regression
from sklearn.metrics import accuracy_score
print('Accuracy Score of Logistic Regression:')
print (accuracy_score (y_test,y_pred))
