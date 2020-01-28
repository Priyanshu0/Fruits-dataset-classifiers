#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 01:59:41 2019

@author: priyanshutuli
"""
import pandas as pd
import matplotlib.pyplot as plt

#importing fruits dataset txt file
fruits = pd.read_table('Downloads/Machine-Learning-with-Python-master/fruit_data_with_colors.txt')
#printing the first 10 rows of the dataset
print(fruits.head(10))
#printing the shape of the dataset
print(fruits.shape)
#the different types of fruits
print(fruits['fruit_name'].unique())
#describing the dataframe
print(fruits.describe())
#the number of fruits of each fruit type
print(fruits.groupby('fruit_name').size())

import seaborn as sns
sns.countplot(fruits['fruit_name'],label="Count")
plt.show()

feature_names = ['mass', 'width', 'height', 'color_score']
#creating feature dataset
X = fruits[feature_names]
#creating output dataset
y = fruits['fruit_label']
#splitting into test and train dataset
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.25, random_state=0)
#applying feature scaling using min max scaling for values btw 0 and 1
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
#applying logistic regression classifier
from sklearn.linear_model import LogisticRegression
#creating object of Logistic
l_reg = LogisticRegression()
l_reg.fit(X_train, y_train)
#printing the accuracy
print('Accuracy of Logistic regression classifier on training set: {:.2f}'
     .format(l_reg.score(X_train, y_train)))
print('Accuracy of Logistic regression classifier on test set: {:.2f}'
     .format(l_reg.score(X_test, y_test)))

#now checking the accuracy through decision tree
from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier().fit(X_train, y_train)
#accuracy from decision tree classifier
print('Accuracy of Decision Tree classifier on training set: {:.2f}'
     .format(dtc.score(X_train, y_train)))
print('Accuracy of Decision Tree classifier on test set: {:.2f}'
     .format(dtc.score(X_test, y_test)))

#applying classification using k nearest neighbours
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
print('Accuracy of K-NN classifier on training set: {:.2f}'
     .format(knn.score(X_train, y_train)))
print('Accuracy of K-NN classifier on test set: {:.2f}'
     .format(knn.score(X_test, y_test)))

#appyling naive bayes
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(X_train, y_train)
print('Accuracy of GNB classifier on training set: {:.2f}'
     .format(gnb.score(X_train, y_train)))
print('Accuracy of GNB classifier on test set: {:.2f}'
     .format(gnb.score(X_test, y_test)))

#applying svm
from sklearn.svm import SVC
svm = SVC()
svm.fit(X_train, y_train)
print('Accuracy of SVM classifier on training set: {:.2f}'
     .format(svm.score(X_train, y_train)))
print('Accuracy of SVM classifier on test set: {:.2f}'
     .format(svm.score(X_test, y_test)))

#decision tree classifier had the highest accuracy
#building confusion matrix for this
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
pred = dtc.predict(X_test)
print(confusion_matrix(y_test, pred))
print(classification_report(y_test, pred))