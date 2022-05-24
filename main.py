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
