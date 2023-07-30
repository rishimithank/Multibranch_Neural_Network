#!/usr/bin/env python
# coding: utf-8

# In[4]:


import torch
import torch.nn as nn
import pandas as pd
# Define the dataset
data=pd.read_csv('D:\Engineering\Machine learning\Mini Project\Antenna\Datasets\Antenna.csv')
dataset = torch.tensor(data.values)

# Define the neural network
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(3, 2)
        self.fc2 = nn.Linear(2, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.sigmoid(x)
        x = self.fc2(x)
        return x

net = Net()

# Check monotonicity of the dataset
for name, param in net.named_parameters():
    gradients_output = torch.autograd.grad(outputs=net(dataset.float()).sum(), inputs=param, create_graph=True)
    if gradients_output[0].ge(torch.tensor(0.0)).all():
        print(f"The network is monotonic with respect to {name} and the directional derivative is always positive.")
    elif gradients_output[0].le(torch.tensor(0.0)).all():
        print(f"The network is monotonic with respect to {name} and the directional derivative is always negative.")
    else:
        print(f"The network is not monotonic with respect to {name}.")


# In[ ]:





# In[ ]:




