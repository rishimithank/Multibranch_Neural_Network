#!/usr/bin/env python
# coding: utf-8

# In[25]:


import numpy as np
import pandas as pd
from tensorflow.keras.layers import Input, Dense, Concatenate
from tensorflow.keras.models import Model


# In[26]:


data=pd.read_csv('D:\Engineering\Machine learning\Mini Project\Antenna\Datasets\Antenna.csv')
data


# In[27]:


X=data[['Directivity(dB)']].values
Y=data[['Spacing(mm)','Phase angle']].values


# In[28]:


x1=X[:360]
x2=X[361:721]
x3=X[722:1082]
x4=X[1083:1443]
x5=X[1444:1804]
x6=X[1805:2165]
x7=X[2166:2526]
x8=X[2527:2887]
x9=X[2888:3248]
x10=X[3249:3609]
x11=X[3610:3970]
x12=X[3971:4331]
x13=X[4332:4692]
x14=X[4693:5053]
x15=X[5054:5414]
x16=X[5415:5775]
x17=X[5776:6136]
x18=X[6137:6497]
x19=X[6498:6858]
x20=X[6859:7219]
x21=X[7220:7580]
x22=X[7581:7941]
x23=X[7942:8302]
x24=X[8303:8663]
x25=X[8664:9024]
x26=X[9025:9385]
x27=X[9386:9746]
x28=X[9747:10107]
x29=X[10108:10468]
x30=X[10469:10829]
x31=X[10830:11190]
x32=X[11191:11551]
x33=X[11552:11912]
x34=X[11913:12273]
x35=X[12274:12634]
x36=X[12635:12995]
x37=X[12996:13356]
#x38=X[13357:13717]


# In[5]:


y1=Y[:360]
y2=Y[361:721]
y3=Y[722:1082]
y4=Y[1083:1443]
y5=Y[1444:1804]
y6=Y[1805:2165]
y7=Y[2166:2526]
y8=Y[2527:2887]
y9=Y[2888:3248]
y10=Y[3249:3609]
y11=Y[3610:3970]
y12=Y[3971:4331]
y13=Y[4332:4692]
y14=Y[4693:5053]
y15=Y[5054:5414]
y16=Y[5415:5775]
y17=Y[5776:6136]
y18=Y[6137:6497]
y19=Y[6498:6858]
y20=Y[6859:7219]
y21=Y[7220:7580]
y22=Y[7581:7941]
y23=Y[7942:8302]
y24=Y[8303:8663]
y25=Y[8664:9024]
y26=Y[9025:9385]
y27=Y[9386:9746]
y28=Y[9747:10107]
y29=Y[10108:10468]
y30=Y[10469:10829]
y31=Y[10830:11190]
y32=Y[11191:11551]
y33=Y[11552:11912]
y34=Y[11913:12273]
y35=Y[12274:12634]
y36=Y[12635:12995]
y37=Y[12996:13356]
#y38=Y[13357:13717]


# In[29]:


split_ratio = 0.8
split_idx = int(len(y1) * split_ratio)


# In[ ]:





# In[30]:


train_input_1 = x1[:split_idx]
train_input_2 = x2[:split_idx]
train_input_3 = x3[:split_idx]
train_input_4 = x4[:split_idx]
train_input_5 = x5[:split_idx]
train_input_6 = x6[:split_idx]
train_input_7 = x7[:split_idx]
train_input_8 = x8[:split_idx]
train_input_9 = x9[:split_idx]
train_input_10 =x10[:split_idx]
train_input_11= x11[:split_idx]
train_input_12= x12[:split_idx]
train_input_13= x13[:split_idx]
train_input_14= x14[:split_idx]
train_input_15= x15[:split_idx]
train_input_16= x16[:split_idx]
train_input_17= x17[:split_idx]
train_input_18= x18[:split_idx]
train_input_19= x19[:split_idx]
train_input_20= x20[:split_idx]
train_input_21= x21[:split_idx]
train_input_22= x22[:split_idx]
train_input_23= x23[:split_idx]
train_input_24= x24[:split_idx]
train_input_25= x25[:split_idx]
train_input_26= x26[:split_idx]
train_input_27= x27[:split_idx]
train_input_28= x28[:split_idx]
train_input_29= x29[:split_idx]
train_input_30= x30[:split_idx]
train_input_31= x31[:split_idx]
train_input_32= x32[:split_idx]
train_input_33= x33[:split_idx]
train_input_34= x34[:split_idx]
train_input_35= x35[:split_idx]
train_input_36= x36[:split_idx]
train_input_37= x37[:split_idx]
#train_input_38= x38[:split_idx]


# In[31]:


train_output_1 = y1[:split_idx]
train_output_2 = y2[:split_idx]
train_output_3 = y3[:split_idx]
train_output_4 = y4[:split_idx]
train_output_5 = y5[:split_idx]
train_output_6 = y6[:split_idx]
train_output_7 = y7[:split_idx]
train_output_8 = y8[:split_idx]
train_output_9 = y9[:split_idx]
train_output_10= y10[:split_idx]
train_output_11= y11[:split_idx]
train_output_12= y12[:split_idx]
train_output_13= y13[:split_idx]
train_output_14= y14[:split_idx]
train_output_15= y15[:split_idx]
train_output_16= y16[:split_idx]
train_output_17= y17[:split_idx]
train_output_18= y18[:split_idx]
train_output_19= y19[:split_idx]
train_output_20= y20[:split_idx]
train_output_21= y21[:split_idx]
train_output_22= y22[:split_idx]
train_output_23= y23[:split_idx]
train_output_24= y24[:split_idx]
train_output_25= y25[:split_idx]
train_output_26= y26[:split_idx]
train_output_27= y27[:split_idx]
train_output_28= y28[:split_idx]
train_output_29= y29[:split_idx]
train_output_30= y30[:split_idx]
train_output_31= y31[:split_idx]
train_output_32= y32[:split_idx]
train_output_33= y33[:split_idx]
train_output_34= y34[:split_idx]
train_output_35= y35[:split_idx]
train_output_36= y36[:split_idx]
train_output_37= y37[:split_idx]
#train_output_38= y38[:split_idx]


# In[32]:


test_input_1 = x1[split_idx:]
test_input_2 = x2[split_idx:]
test_input_3 = x3[split_idx:]
test_input_4 = x4[split_idx:]
test_input_5 = x5[split_idx:]
test_input_6 = x6[split_idx:]
test_input_7 = x7[split_idx:]
test_input_8 = x8[split_idx:]
test_input_9 = x9[split_idx:]
test_input_10= x10[split_idx:]
test_input_11= x11[split_idx:]
test_input_12= x12[split_idx:]
test_input_13= x13[split_idx:]
test_input_14= x14[split_idx:]
test_input_15= x15[split_idx:]
test_input_16= x16[split_idx:]
test_input_17= x17[split_idx:]
test_input_18= x18[split_idx:]
test_input_19= x19[split_idx:]
test_input_20= x20[split_idx:]
test_input_21= x21[split_idx:]
test_input_22= x22[split_idx:]
test_input_23= x23[split_idx:]
test_input_24= x24[split_idx:]
test_input_25= x25[split_idx:]
test_input_26= x26[split_idx:]
test_input_27= x27[split_idx:]
test_input_28= x28[split_idx:]
test_input_29= x29[split_idx:]
test_input_30= x30[split_idx:]
test_input_31= x31[split_idx:]
test_input_32= x32[split_idx:]
test_input_33= x33[split_idx:]
test_input_34= x34[split_idx:]
test_input_35= x35[split_idx:]
test_input_36= x36[split_idx:]
test_input_37= x37[split_idx:]
#test_input_38= x38[split_idx:]


# In[33]:


test_output_1 = y1[split_idx:]
test_output_2 = y2[split_idx:]
test_output_3 = y3[split_idx:]
test_output_4 = y4[split_idx:]
test_output_5 = y5[split_idx:]
test_output_6 = y6[split_idx:]
test_output_7 = y7[split_idx:]
test_output_8 = y8[split_idx:]
test_output_9 = y9[split_idx:]
test_output_10= y10[split_idx:]
test_output_11= y11[split_idx:]
test_output_12= y12[split_idx:]
test_output_13= y13[split_idx:]
test_output_14= y14[split_idx:]
test_output_15= y15[split_idx:]
test_output_16= y16[split_idx:]
test_output_17= y17[split_idx:]
test_output_18= y18[split_idx:]
test_output_19= y19[split_idx:]
test_output_20= y20[split_idx:]
test_output_21= y21[split_idx:]
test_output_22= y22[split_idx:]
test_output_23= y23[split_idx:]
test_output_24= y24[split_idx:]
test_output_25= y25[split_idx:]
test_output_26= y26[split_idx:]
test_output_27= y27[split_idx:]
test_output_28= y28[split_idx:]
test_output_29= y29[split_idx:]
test_output_30= y30[split_idx:]
test_output_31= y31[split_idx:]
test_output_32= y32[split_idx:]
test_output_33= y33[split_idx:]
test_output_34= y34[split_idx:]
test_output_35= y35[split_idx:]
test_output_36= y36[split_idx:]
test_output_37= y37[split_idx:]
#test_output_38= y38[split_idx:]


# In[34]:


input_1 = Input(shape=(1,))
input_2 = Input(shape=(1,))
input_3 = Input(shape=(1,))
input_4 = Input(shape=(1,))
input_5 = Input(shape=(1,))
input_6 = Input(shape=(1,))
input_7 = Input(shape=(1,))
input_8 = Input(shape=(1,))
input_9 = Input(shape=(1,))
input_10= Input(shape=(1,))
input_11= Input(shape=(1,))
input_12= Input(shape=(1,))
input_13= Input(shape=(1,))
input_14= Input(shape=(1,))
input_15= Input(shape=(1,))
input_16= Input(shape=(1,))
input_17= Input(shape=(1,))
input_18= Input(shape=(1,))
input_19= Input(shape=(1,))
input_20= Input(shape=(1,))
input_21= Input(shape=(1,))
input_22= Input(shape=(1,))
input_23= Input(shape=(1,))
input_24= Input(shape=(1,))
input_25= Input(shape=(1,))
input_26= Input(shape=(1,))
input_27= Input(shape=(1,))
input_28= Input(shape=(1,))
input_29= Input(shape=(1,))
input_30= Input(shape=(1,))
input_31= Input(shape=(1,))
input_32= Input(shape=(1,))
input_33= Input(shape=(1,))
input_34= Input(shape=(1,))
input_35= Input(shape=(1,))
input_36= Input(shape=(1,))
input_37= Input(shape=(1,))
#input_38= Input(shape=(360,))


# In[35]:


dense_1 = Dense(64, activation='relu')(input_1)
dense_2 = Dense(64, activation='relu')(input_2)
dense_3 = Dense(64, activation='relu')(input_3)
dense_4 = Dense(64, activation='relu')(input_4)
dense_5 = Dense(64, activation='relu')(input_5)
dense_6 = Dense(64, activation='relu')(input_6)
dense_7 = Dense(64, activation='relu')(input_7)
dense_8 = Dense(64, activation='relu')(input_8)
dense_9 = Dense(64, activation='relu')(input_9)
dense_10= Dense(64, activation='relu')(input_10)
dense_11= Dense(64, activation='relu')(input_11)
dense_12= Dense(64, activation='relu')(input_12)
dense_13= Dense(64, activation='relu')(input_13)
dense_14= Dense(64, activation='relu')(input_14)
dense_15= Dense(64, activation='relu')(input_15)
dense_16= Dense(64, activation='relu')(input_16)
dense_17= Dense(64, activation='relu')(input_17)
dense_18= Dense(64, activation='relu')(input_18)
dense_19= Dense(64, activation='relu')(input_19)
dense_20= Dense(64, activation='relu')(input_20)
dense_21= Dense(64, activation='relu')(input_21)
dense_22= Dense(64, activation='relu')(input_22)
dense_23= Dense(64, activation='relu')(input_23)
dense_24= Dense(64, activation='relu')(input_24)
dense_25= Dense(64, activation='relu')(input_25)
dense_26= Dense(64, activation='relu')(input_26)
dense_27= Dense(64, activation='relu')(input_27)
dense_28= Dense(64, activation='relu')(input_28)
dense_29= Dense(64, activation='relu')(input_29)
dense_30= Dense(64, activation='relu')(input_30)
dense_31= Dense(64, activation='relu')(input_31)
dense_32= Dense(64, activation='relu')(input_32)
dense_33= Dense(64, activation='relu')(input_33)
dense_34= Dense(64, activation='relu')(input_34)
dense_35= Dense(64, activation='relu')(input_35)
dense_36= Dense(64, activation='relu')(input_36)
dense_37= Dense(64, activation='relu')(input_37)
#dense_38= Dense(64, activation='relu')(input_38)


# In[36]:


concat = Concatenate()([dense_1, dense_2,dense_3,dense_4,dense_5,dense_6,dense_7,dense_8,dense_9,dense_10,dense_11,dense_12,dense_13,dense_14,dense_15,dense_16,dense_17,dense_18,dense_19,dense_20,dense_21,dense_22,dense_23,dense_24,dense_25,dense_26,dense_27,dense_28,dense_29,dense_30,dense_31,dense_32,dense_33,dense_34,dense_35,dense_36,dense_37])


# In[37]:


output_1 = Dense(2, activation='softmax', name='output_1')(concat)
output_2 = Dense(2, activation='softmax', name='output_2')(concat)
output_3 = Dense(2, activation='softmax', name='output_3')(concat)
output_4 = Dense(2, activation='softmax', name='output_4')(concat)
output_5 = Dense(2, activation='softmax', name='output_5')(concat)
output_6 = Dense(2, activation='softmax', name='output_6')(concat)
output_7 = Dense(2, activation='softmax', name='output_7')(concat)
output_8 = Dense(2, activation='softmax', name='output_8')(concat)
output_9 = Dense(2, activation='softmax', name='output_9')(concat)
output_10= Dense(2, activation='softmax', name='output_10')(concat)
output_11= Dense(2, activation='softmax', name='output_11')(concat)
output_12= Dense(2, activation='softmax', name='output_12')(concat)
output_13= Dense(2, activation='softmax', name='output_13')(concat)
output_14= Dense(2, activation='softmax', name='output_14')(concat)
output_15= Dense(2, activation='softmax', name='output_15')(concat)
output_16= Dense(2, activation='softmax', name='output_16')(concat)
output_17= Dense(2, activation='softmax', name='output_17')(concat)
output_18= Dense(2, activation='softmax', name='output_18')(concat)
output_19= Dense(2, activation='softmax', name='output_19')(concat)
output_20= Dense(2, activation='softmax', name='output_20')(concat)
output_21= Dense(2, activation='softmax', name='output_21')(concat)
output_22= Dense(2, activation='softmax', name='output_22')(concat)
output_23= Dense(2, activation='softmax', name='output_23')(concat)
output_24= Dense(2, activation='softmax', name='output_24')(concat)
output_25= Dense(2, activation='softmax', name='output_25')(concat)
output_26= Dense(2, activation='softmax', name='output_26')(concat)
output_27= Dense(2, activation='softmax', name='output_27')(concat)
output_28= Dense(2, activation='softmax', name='output_28')(concat)
output_29= Dense(2, activation='softmax', name='output_29')(concat)
output_30= Dense(2, activation='softmax', name='output_30')(concat)
output_31= Dense(2, activation='softmax', name='output_31')(concat)
output_32= Dense(2, activation='softmax', name='output_32')(concat)
output_33= Dense(2, activation='softmax', name='output_33')(concat)
output_34= Dense(2, activation='softmax', name='output_34')(concat)
output_35= Dense(2, activation='softmax', name='output_35')(concat)
output_36= Dense(2, activation='softmax', name='output_36')(concat)
output_37= Dense(2, activation='softmax', name='output_37')(concat)
#output_38= Dense(10, activation='softmax', name='output_38')(concat)


# In[38]:


model = Model(inputs=[input_1, input_2,input_3,input_4,input_5,input_6,input_7,input_8,input_9,input_10,input_11,input_12,input_13,input_14,input_15,input_16,input_17,input_18,input_19,input_20,input_21,input_22,input_23,input_24,input_25,input_26,input_27,input_28,input_29,input_30,input_31,input_32,input_33,input_34,input_35,input_36,input_37], outputs=[output_1, output_2,output_3,output_4,output_5,output_6,output_7,output_8,output_9,output_10,output_11,output_12,output_13,output_14,output_15,output_16,output_17,output_18,output_19,output_20,output_21,output_22,output_23,output_24,output_25,output_26,output_27,output_28,output_29,output_30,output_31,output_32,output_33,output_34,output_35,output_36,output_37])


# In[39]:


model.compile(loss={'output_1': 'mse', 'output_2': 'mse','output_3': 'mse','output_4': 'mse','output_5': 'mse','output_6': 'mse','output_7': 'mse','output_8': 'mse','output_9': 'mse','output_10': 'mse','output_11': 'mse','output_12': 'mse','output_13': 'mse','output_14': 'mse','output_15': 'mse','output_16': 'mse','output_17': 'mse','output_18': 'mse','output_19': 'mse','output_20': 'mse','output_21': 'mse','output_22': 'mse','output_23': 'mse','output_24': 'mse','output_25': 'mse','output_26': 'mse','output_27': 'mse','output_28': 'mse','output_29': 'mse','output_30': 'mse','output_31': 'mse','output_32': 'mse','output_33': 'mse','output_34': 'mse','output_35': 'mse','output_36': 'mse','output_37': 'mse'},
              optimizer='adam')


# In[40]:


model.fit([train_input_1, train_input_2,train_input_3,train_input_4,train_input_5,train_input_6,train_input_7,train_input_8,train_input_9,train_input_10,train_input_11,train_input_12,train_input_13,train_input_14,train_input_15,train_input_16,train_input_17,train_input_18,train_input_19,train_input_20,train_input_21,train_input_22,train_input_23,train_input_24,train_input_25,train_input_26,train_input_27,train_input_28,train_input_29,train_input_30,train_input_31,train_input_32,train_input_33,train_input_34,train_input_35,train_input_36,train_input_37], [train_output_1, train_output_2,train_output_3,train_output_4,train_output_5,train_output_6,train_output_7,train_output_8,train_output_9,train_output_10,train_output_11,train_output_12,train_output_13,train_output_14,train_output_15,train_output_16,train_output_17,train_output_18,train_output_19,train_output_20,train_output_21,train_output_22,train_output_23,train_output_24,train_output_25,train_output_26,train_output_27,train_output_28,train_output_29,train_output_30,train_output_31,train_output_32,train_output_33,train_output_34,train_output_35,train_output_36,train_output_37],
          validation_data=([test_input_1,test_input_2,test_input_3,test_input_4,test_input_5,test_input_6,test_input_7,test_input_8,test_input_9,test_input_10,test_input_11,test_input_12,test_input_13,test_input_14,test_input_15,test_input_16,test_input_17,test_input_18,test_input_19,test_input_20,test_input_21,test_input_22,test_input_23,test_input_24,test_input_25,test_input_26,test_input_27,test_input_28,test_input_29,test_input_30,test_input_31,test_input_32,test_input_33,test_input_34,test_input_35,test_input_36,test_input_37], [test_output_1, test_output_2,test_output_3,test_output_4,test_output_5,test_output_6,test_output_7,test_output_8,test_output_9,test_output_10,test_output_11,test_output_12,test_output_13,test_output_14,test_output_15,test_output_16,test_output_17,test_output_18,test_output_19,test_output_20,test_output_21,test_output_22,test_output_23,test_output_24,test_output_25,test_output_26,test_output_27,test_output_28,test_output_29,test_output_30,test_output_31,test_output_32,test_output_33,test_output_34,test_output_35,test_output_36,test_output_37]),
          epochs=10, batch_size=32)


# In[43]:


acc = model.evaluate([test_input_1,test_input_2,test_input_3,test_input_4,test_input_5,test_input_6,test_input_7,test_input_8,test_input_9,test_input_10,test_input_11,test_input_12,test_input_13,test_input_14,test_input_15,test_input_16,test_input_17,test_input_18,test_input_19,test_input_20,test_input_21,test_input_22,test_input_23,test_input_24,test_input_25,test_input_26,test_input_27,test_input_28,test_input_29,test_input_30,test_input_31,test_input_32,test_input_33,test_input_34,test_input_35,test_input_36,test_input_37], [test_output_1, test_output_2,test_output_3,test_output_4,test_output_5,test_output_6,test_output_7,test_output_8,test_output_9,test_output_10,test_output_11,test_output_12,test_output_13,test_output_14,test_output_15,test_output_16,test_output_17,test_output_18,test_output_19,test_output_20,test_output_21,test_output_22,test_output_23,test_output_24,test_output_25,test_output_26,test_output_27,test_output_28,test_output_29,test_output_30,test_output_31,test_output_32,test_output_33,test_output_34,test_output_35,test_output_36,test_output_37])


# In[ ]:





# In[ ]:




