#!/usr/bin/env python
# coding: utf-8

# In[8]:


import pandas as pd
import numpy as np
import seaborn as sns
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import r2_score
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# In[2]:


data=pd.read_csv('D:\Machine learning\Mini Project\Antenna\Datasets\Antenna.csv')
data


# In[3]:


X=np.asarray(data.drop(columns='Directivity(dB)',axis=1))
Y=np.asarray(data[['Directivity(dB)']])


# In[12]:


def ann_model():
    model=Sequential()
    model.add(Dense(32,input_dim=2,activation='relu'))
    model.add(Dense(16,activation='relu'))
    model.add(Dense(8,activation='relu'))
    model.add(Dense(1))
    model.compile(loss='mse',optimizer='adam')
    return model


# In[13]:


x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.2,random_state=0)


# In[14]:


std=StandardScaler()
x_train=std.fit_transform(x_train)
x_test=std.fit_transform(x_test)


# In[65]:


estimator = KerasRegressor(build_fn=ann_model, epochs=30, batch_size=1, verbose=1)


# In[66]:


estimator.fit(x_train,y_train)


# In[67]:


pred=estimator.predict(x_test)


# In[73]:


error=np.abs(pred-y_test)
tot=np.mean(error)
print('mean error =',round(tot,2))


# In[70]:


train_pred=estimator.predict(x_train)
print('accuracy =',round(r2_score(train_pred,y_train)*100,2))


# In[ ]:




