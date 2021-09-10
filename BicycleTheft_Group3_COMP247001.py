# -*- coding: utf-8 -*-
"""
Created on Fri Jul 16 11:11:26 2021

@author: Divya
"""

import pandas as pd
import os
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.utils import resample
from sklearn.metrics import accuracy_score,precision_score,recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
import joblib
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import RandomizedSearchCV


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
df2 = df["Status"]
df2.hist()
plt.show()


#Scatter Plot
sns.pairplot(df)

#Heat Map
sns.heatmap(df.corr(), cmap='coolwarm')

## dropping unnecessary columns
df.drop(['X','Y','OBJECTID','event_unique_id', 'ObjectId2'],inplace = True, axis=1)


## applyimg label encoding for converting categorical data to numric values
from sklearn.preprocessing import LabelEncoder
label_enc = LabelEncoder()

df = df.apply(lambda col: label_enc.fit_transform(col.astype(str)), axis=0, result_type='expand')

df['Status'].plot(kind='box')
plt.show()
df['Status'].plot(kind='hist')
plt.show()

#removing rows with unknown values in the Status class 
df = df[df["Status"]!=2]

## splitting into features and target
df_features=df[df.columns.difference(['Status'])] 
df_target=df['Status'] 

## balance the imablanced classes

df['Status'].value_counts()
df_majority = df[df.Status==1]
df_minority = df[df.Status==0]
df_features_upsampled = resample(df_minority, 
                                 replace=True,     # sample with replacement
                                 n_samples=24807,    # to match majority class                               
                                 random_state=123)
# Combine majority class with upsampled features class
df_upsampled = pd.concat([df_majority,df_features_upsampled])
 
# Display new class counts
df_upsampled.Status.value_counts()

y = df_upsampled.Status
x = df_upsampled.drop('Status', axis=1)

## splitting numeric and categorical features
numeric_features=['Occurrence_Year','Occurrence_DayOfMonth','Occurrence_DayOfYear','Occurrence_Hour','Report_Year', 'Report_DayOfMonth',
                  'Report_DayOfYear', 'Bike_Speed', 'Cost_of_Bike', 'Report_Hour']

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

x_transformed = full_pipeline.fit_transform(x)


## splitting data into train and test
x_train,x_test,y_train,y_test = train_test_split(x_transformed, y, test_size = 0.35)

### Building models 

def model_results(y_pred, y_test, x_test, model, estimator):
    
    # Evaluation:
    print("Estimator: ",estimator)
    # Accuracy score:
    print("Accuracy:", accuracy_score(y_test, y_pred)) 
    print("Precision score:",precision_score(y_test, y_pred))
    print("Recall score:",recall_score(y_test, y_pred))
    print("F1 score:", f1_score(y_test, y_pred, average = "micro"), "\n")
    metrics.plot_roc_curve(model, x_test, y_test)  
    plt.show() 
    cm = (confusion_matrix(y_test,y_pred))
    sns.heatmap(cm,annot=True,fmt='g')
    

lr = LogisticRegression(max_iter = 1600,random_state=42)
lr.fit(x_train,y_train)
y_pred_lr = lr.predict(x_test)
model_results(y_pred_lr, y_test, x_test, lr, 'Logistic regression')
scores = cross_val_score(lr, x_train, y_train, cv=5)
print("Cross validation Score: ", scores.mean())

rfc = RandomForestClassifier()
rfc.fit(x_train,y_train)
y_pred_rfc = rfc.predict(x_test)
model_results(y_pred_rfc, y_test, x_test, rfc, 'Random Forest')
scores = cross_val_score(rfc, x_train, y_train, cv=5)
print("Cross validation Score: ", scores.mean())

svm = SVC(C=0.1, kernel='linear')
svm.fit(x_train,y_train)
y_pred_svm = svm.predict(x_test)
model_results(y_pred_svm, y_test, x_test, svm, 'Support Vector Machine')
scores = cross_val_score(svm, x_train, y_train, cv=5)
print("Cross validation Score: ", scores.mean())

dtc = DecisionTreeClassifier(criterion='entropy', max_depth=42)
dtc.fit(x_train,y_train)
y_pred_dtc = dtc.predict(x_test)
model_results(y_pred_dtc, y_test, x_test, dtc, 'Decision Tree')
scores = cross_val_score(dtc, x_train, y_train, cv=5)
print("Cross validation Score: ", scores.mean())

mlp = MLPClassifier(random_state=42)
mlp.fit(x_train,y_train)
y_pred_mlp = mlp.predict(x_test)
model_results(y_pred_mlp, y_test, x_test, mlp, 'MLP')
scores = cross_val_score(mlp, x_train, y_train, cv=5)
print("Cross validation Score: ", scores.mean())

gb = GradientBoostingClassifier(n_estimators=20, max_features=2, max_depth=2, random_state=42)
gb.fit(x_train, y_train)
y_pred_gb = gb.predict(x_test)
model_results(y_pred_gb, y_test, x_test, gb, 'Gradient Boosting Classifier')
scores = cross_val_score(gb, x_train, y_train, cv=5)
print("Cross validation Score: ", scores.mean())

#########Ensemble using voting#############
voting_clf = VotingClassifier(
    estimators=[('lr', lr), ('rf', rfc), ('dt', dtc), ('svm', svm),
                ('mlp', mlp)],
    voting='hard')

voting_clf.fit(x_train, y_train.values.ravel())

for clf in (lr, rfc, dtc, svm, mlp, voting_clf):
    clf.fit(x_train, y_train.values.ravel())
    y_pred_hard = clf.predict(x_test)
    print(clf, accuracy_score(y_test.values.ravel(), y_pred_hard))
    print('\n')
 

##########tuning with adaboosting###########
    
#from sklearn.ensemble import AdaBoostClassifier
#adb_clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1),random_state=42)
#adb_clf.fit(x_train,y_train)
#y_pred = adb_clf.predict(x_test)
#print('Accuracy: ', round(accuracy_score(y_test, y_pred), 2))
#print('f1_score: ', round(f1_score(y_test, y_pred), 2))
#print('precision: ', round(precision_score(y_test, y_pred), 2))
#print('recall: ', round(recall_score(y_test, y_pred), 2))

#cross validation\r\n",

#scores = cross_val_score(adb_clf,x_train,y_train,cv=10,scoring='accuracy')
#print(scores.mean())
#params_adb = {'n_estimators':range(1,200),'learning_rate':[0.1,0.3,0.5,0.7,0.9]}
#randomized_adb = RandomizedSearchCV(
 #                   estimator=adb_clf,
  #                  param_distributions=params_adb,
   #                 scoring='recall',
    #                n_jobs=-1,
     #               cv=10,
      #              refit=True,
       #             verbose=3)

#randomized_adb.fit(x_train,y_train)
#adb_best = randomized_adb.best_estimator_
#y_pred = adb_best.predict(x_test)
#cm = confusion_matrix(y_test,y_pred)
#sns.heatmap(cm,annot=True,fmt='g')

#print('f1_score of LR: ', round(f1_score(y_test, y_pred), 2)),
#print('precision of LR: ', round(precision_score(y_test, y_pred), 2))
#print('recall of LR: ', round(recall_score(y_test, y_pred), 2))
#y_pred_thresold = adb_best.predict_proba(x_test)


##########Grid Search Cv tuning##########
#params = {'n_estimators':[50,100],
#            'criterion':["gini","entropy"],
 #           'max_leaf_nodes':[40, 60],
  #          'min_samples_split':[5, 10, 20],
   #         'max_features':[20,40, 60]}
          
#grid_search = GridSearchCV(
 #                   estimator=rfc,
   #                 param_grid=params,
   #                 scoring='accuracy',
   #                 cv=5,
   #                 refit=True,
   #                 verbose=3
#)



############RandomisedSearchCV tuning##################
params = {'n_estimators':range(0,100),
            'criterion':["gini","entropy"],
            'max_depth':range(1,30),
            'min_samples_split':[0.1,0.3,0.5,0.7,0.9],
            'max_features':['auto','sqrt','log2']}
random_search = RandomizedSearchCV(
                    estimator=rfc,
                    param_distributions=params,
                    n_iter = 30,
                    scoring='accuracy',
                    n_jobs=-1,
                    cv=10,
                    refit=True,
                    verbose=3
)

random_search.fit(x_train, y_train)
print(random_search.best_params_)
rf_best = random_search.best_estimator_
print(rf_best)

y_rf_best_pred = rf_best.predict(x_test)

predict_proba_test = rf_best.predict_proba(x_test)
predict_proba_test = predict_proba_test[:,1]
print(predict_proba_test)

import pickle
# save the model to disk
#filename = 'finalized_model.pkl'
pickle.dump(random_search, open('finalized_model.sav', 'wb'))

joblib.dump(random_search, filename)
# save the pipeline to disk
pickle.dump(full_pipeline, open('finalized_pipeline.sav', 'wb'))

#filename_pipeline = 'finalized_pipeline.pkl'
#joblib.dump(full_pipeline, filename_pipeline)

loaded_model = joblib.load(filename)
result = loaded_model.score(x_test, y_test)
print(result)


