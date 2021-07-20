# -*- coding: utf-8 -*-
"""
Created on Fri Jul 16 11:11:26 2021

@author: Divya
"""

import pandas as pd
import os
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer


# Loading the file
filename = 'Bicycle_Thefts.csv'
path = 'C:/Users/Divya/Documents/Centennial/Sem2/Supervised Learning'
fullpath = os.path.join(path,filename)
df = pd.read_csv(fullpath)

df.head()
df.info()
df.columns

## range of data elements
df.describe().max() - df.describe().min()

## description
df.describe()

## finding null values
df.isnull().sum()

## plottimg histogram
df.hist(figsize=(10,12))
plt.show()

#Scatter Plot
sns.pairplot(df);

#Heat Map
sns.heatmap(df.corr(), cmap='coolwarm')

## dropping unnecessary columns
df.drop(['X','Y','OBJECTID','event_unique_id', 'ObjectId2', 'Report_Hour'],inplace = True, axis=1)

#removing rows with unknown values in the Status class 
df = df[df["Status"]!="UNKNOWN"]

## applyimg label encoding for converting categorical data to numric values
from sklearn.preprocessing import LabelEncoder
label_enc = LabelEncoder()

df = df.apply(lambda col: label_enc.fit_transform(col.astype(str)), axis=0, result_type='expand')

## splitting into features and target
df_features=df[df.columns.difference(['Status'])] 
df_target=df['Status'] 


## splitting data into train and test
x_train,x_test,y_train,y_test = train_test_split(df_features, df_target, test_size = 0.35)


## splitting numeric and categorical features
numeric_features=['Occurrence_Year','Occurrence_DayOfMonth','Occurrence_DayOfYear','Occurrence_Hour','Report_Year', 'Report_DayOfMonth',
                  'Report_DayOfYear', 'Bike_Speed', 'Cost_of_Bike']

cat_features = []
for i in df_features.columns:
    if i not in numeric_features:
        cat_features.append(i)

## creating pipeline
numeric_pipeline = Pipeline([
    ('impute',SimpleImputer(strategy='median')),
    ('standardization',StandardScaler())
])
category_pipeline = Pipeline([
    ('imputer',SimpleImputer(strategy='most_frequent')),
    ('OneHotEncoding',OneHotEncoder())
])

full_pipeline = ColumnTransformer([
    ('num_transformations',numeric_pipeline,numeric_features),
    ('cat_transformations',category_pipeline,cat_features)
])

x_train_transformed = full_pipeline.fit_transform(x_train)

x_test_transformed = full_pipeline.fit_transform(x_test)


