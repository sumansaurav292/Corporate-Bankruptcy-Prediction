# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 23:02:08 2019

@author: dell
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier

from sklearn import metrics

df=pd.read_csv("D:\\Torrent downloads\\data\\5year.csv",low_memory=False)
clss=list(df.iloc[:,64])
bankrupt=clss.count(0)
servive=len(clss)-bankrupt
        
data=df.iloc[:,1:64]


for i in range(data.shape[1]):
    x1=data.iloc[:,i]
    x1.replace('?',0,inplace=True)
    mo=np.array(x1,dtype=float)
    mn=np.mean(mo)
    x1.replace(0,mn,inplace=True)
    data.iloc[:,i]=x1
     
  
ds=np.array(data.T,dtype=float)
for i in range(len(ds)):
    for j in range(ds.shape[1]):
        sd=np.std(ds[i])
        m=np.mean(ds[i])
        ds[i,j]=(ds[i,j]-m)/sd
        
cleaned_data=pd.DataFrame(ds.T)
x=cleaned_data
y=df.iloc[:,64]


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

"""
clf=RandomForestClassifier(n_estimators=10)
clf.fit(x_train,y_train)
y_pred=clf.predict(x_test)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
"""


"""
clf=GradientBoostingClassifier(n_estimators=100)
clf.fit(x_train,y_train)
y_pred=clf.predict(x_test)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))"""

clf = MLPClassifier(solver='adam', alpha=0.00001, hidden_layer_sizes=(5, 2), random_state=1)
clf.fit(x_train,y_train)
y_pred=clf.predict(x_test)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

confusion=metrics.confusion_matrix(y_test, y_pred)

TP = confusion[1, 1]
TN = confusion[0, 0]
FP = confusion[0, 1]
FN = confusion[1, 0]

Accuracy=(TP + TN) / float(TP + TN + FP + FN)
classification_error = (FP + FN) / float(TP + TN + FP + FN)
sensitivity = TP / float(FN + TP)
specificity = TN / (TN + FP)

false_positive_rate = FP / float(TN + FP)
precision = TP / float(TP + FP)

print(metrics.roc_auc_score(y_test, y_pred))




















       