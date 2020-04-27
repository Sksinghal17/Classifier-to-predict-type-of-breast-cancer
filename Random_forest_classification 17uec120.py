# -*- coding: utf-8 -*-
"""
Created on Sun Mar 22 20:22:29 2020

@author: shubham kumar singhal - 17uec120
"""
#To reduce the overfitting from the decision tree model, we need to prune the tree i.e remove those insignificant
#independent parameters. But that may be a difficult and tedious task. So, we use random forest

#We randomly pick some random parameters and form a decision tree. We make some such trees and find the avg
#of predicted value. This way, overfitting is reduced.

#CLASSIFICATION

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Importing the dataset
dataset = pd.read_csv('data_ss.csv')
X1=dataset.iloc[:,2:35].values
X = dataset.iloc[:, 3:35].values
y = dataset.iloc[:, 1].values



#Removing missing values
from sklearn.impute import SimpleImputer
imputer=SimpleImputer(missing_values=np.nan,strategy="mean",verbose=0)
X[:,3:35]=imputer.fit_transform(X[:,3:35])



# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
#feature scaling is must when our algorithm is based on euclidean distance.
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fitting Random forest classifier to the Training set
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=10,criterion='entropy',random_state=0)
classifier.fit(X_train,y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)



# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
from sklearn.metrics import accuracy_score 
print('Accuracy : ')
print(accuracy_score(y_test,y_pred)) 
from sklearn.metrics import classification_report
print('Report : ')
print(classification_report(y_test,y_pred))

print('confusion matrix : ')
print(cm)


#REGRESSION
dataset = pd.read_csv('data_ss.csv')
dataset=dataset[dataset['Outcome']=='R']

#Feature Selection
#get correlations of each features in dataset
corrmat = dataset.corr()
top_corr_features = corrmat.index
plt.figure(figsize=(30,30))
#plot heat map
g=sns.heatmap(dataset[top_corr_features].corr(),annot=True,cmap="RdYlGn")

X= dataset.iloc[:, 3:35].values
y= dataset.iloc[:, 2].values

#Removing all negative values
dataset.drop([ "Worst_concave_points","fractal_dimension_std_dev","symmetry_std_dev","concavity_std_dev","compactness_std_dev","smoothness_std_dev"],axis=1,inplace=True)

from sklearn.impute import SimpleImputer
imputer=SimpleImputer(missing_values=np.nan,strategy="mean",)
X[:,3:35]=imputer.fit_transform(X[:,3:35])

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

from sklearn.ensemble import RandomForestRegressor
regressor=RandomForestRegressor(n_estimators=1500,random_state=0)
regressor.fit(X_train,y_train)
y_pred=regressor.predict(X_test)

from sklearn.metrics import r2_score
r2_score(y_test,y_pred)
print('r2_score: ')
print(r2_score)