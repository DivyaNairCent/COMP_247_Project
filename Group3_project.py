# -*- coding: utf-8 -*-
"""
Created on Fri Jul 16 11:11:26 2021

@author: Divya
"""

import pandas as pd
import os
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
import joblib
import seaborn as sns
import matplotlib.pyplot as plt


# Loading the file
filename = 'Bicycle_Thefts.csv'
path = 'C:/Users/Divya/Documents/Centennial/Sem2/Supervised Learning'
fullpath = os.path.join(path,filename)
df = pd.read_csv(fullpath)

df.describe()
df.dtypes

df.isnull().sum()

df.dtypes

df.hist(figsize=(10,12))
plt.show()

#Scatter Plot
sns.pairplot(df);

#Heat Map
sns.heatmap(df.corr(), cmap='coolwarm')

## handling missing data
df['Bike_Make'].fillna(df['Bike_Make'].mode(), inplace=True);
df['Bike_Colour'].fillna(df['Bike_Colour'].mode(), inplace=True);
df['Cost_of_Bike'].fillna(df['Cost_of_Bike'].median(), inplace=True);

##categorical columns to numeric values
df_num = pd.get_dummies(df, columns = ['Primary_Offence', 'Report_DayOfWeek', 'Occurrence_Month', 'Occurrence_DayOfWeek','Report_Month','Division','City','Hood_ID','NeighbourhoodName','Location_Type','Premises_Type','Bike_Make','Bike_Model','Bike_Type','Bike_Colour'], drop_first = True)

## dropping unnecessary columns
df_num.drop(['X','Y','OBJECTID','event_unique_id', 'ObjectId2'],inplace = True, axis=1)


