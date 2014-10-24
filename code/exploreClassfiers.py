#! /usr/bin/python
#Import relevant packages
#Pandas for data loading
#Scikit Learn for classification
from sklearn import svm
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.cross_validation import cross_val_score
from sklearn.preprocessing import Imputer

#load in train data set using pandas csv function
train_df = pd.read_csv("../data/training-weka.csv", na_values=['?'], sep=',')
list(train_df.columns.values)

#Train and test a classifier
et = ExtraTreesClassifier(n_estimators=100, max_depth=None, min_samples_split=1, random_state=0)
 
columns = list(train_df.columns.values)
 
labels = train_df["Label"].values
features = train_df[columns[1:31]].values

imp = Imputer(missing_values='NaN', strategy='mean', axis=0, copy=False)
features = imp.fit_transform(features)
 
et_score = cross_val_score(et, features, labels, n_jobs=-1).mean()
 
print("{0} -> ET: {1})".format(columns, et_score))
