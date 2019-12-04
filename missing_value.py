# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 20:51:28 2019

@author: dell
"""

import numpy as np
import pandas as pd
df=pd.read_csv("D:\\Torrent downloads\\data\\1year.csv")
ds=np.array(df)

def separateByLabels(dataset):
	separated = {}
	for i in range(len(dataset)):
		vector = dataset[i]
		if (vector[-1] not in separated):
			separated[vector[-1]] = []
		separated[vector[-1]].append(vector)
	return separated

separated=separateByLabels(ds)
bankrupt_df=separated[1]
nt_bankrupt_df=separated=[0]


nt_bk=pd.DataFrame(nt_bankrupt_df)        #not bankrupt
bk=pd.DataFrame(bankrupt_df)
data=nt_bk.iloc[:,0:64]
data1=bk.iloc[:,0:64]

dic={}
for i in range(data.shape[1]):
    x1=data.iloc[:,i]
    x1=list(x1)
    ml=x1.count('?')
    mk=(ml/len(x1))*100
    dic[i]=mk
    
dic1={} 
for i in range(data1.shape[1]):
    x2=data1.iloc[:,i]
    x2=list(x2)
    mll=x2.count('?')
    mkk=(mll/len(x2))*100
    dic1[i]=mkk
    




   
