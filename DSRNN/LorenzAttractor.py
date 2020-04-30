from __future__ import unicode_literals, print_function, division
from io import open
import glob
import os
import numpy as np
from scipy.integrate import odeint
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

def lorenz(x, y, z, s=10, r=28, b=2.667):
    '''
    Given:
       x, y, z: a point of interest in three dimensional space
       s, r, b: parameters defining the lorenz attractor
    Returns:
       x_dot, y_dot, z_dot: values of the lorenz attractor's partial
           derivatives at the point x, y, z
    adapted from https://matplotlib.org/3.1.0/gallery/mplot3d/lorenz_attractor.html
    '''
    x_dot = s*(y - x)
    y_dot = r*x - y - x*z
    z_dot = x*y - b*z
    return x_dot, y_dot, z_dot

def generatesamp(s0, t_steps):
    """
    returns an array with x, y and z values for every timestep by applying function
    """
    dt=.01
    s = np.empty([t_steps + 1,3])
    s[0] = s0
    for i in range(t_steps):
        x_dot, y_dot, z_dot = lorenz(s[i][0], s[i][1], s[i][2])
        s[i + 1][0] = s[i][0] + (x_dot * dt)
        s[i + 1][1] = s[i][1] + (y_dot * dt)
        s[i + 1][2] = s[i][2] + (z_dot * dt)
    return s

import random

def generate_data(batch_size,t_steps):
    """
    returns a batch_size x t_steps x 3 matrix
    """
    #batch_size number of sequences, each w t_steps data points
    data = np.zeros((batch_size, t_steps,3), dtype =np.float32)
    for i in range(batch_size):
        s0 = [random.uniform(-1,1),random.uniform(-1,1),random.uniform(-1,1)]
        s=generatesamp(s0, t_steps)
        for j in range(t_steps):
            #data populated with x(t) for each t
            data[i][j][0]=s[j][0]
            data[i][j][1]=s[j][1]
            data[i][j][2]=s[j][2]
    return data

def input_and_target(data):
    """
    Generates a matrix of input and target sequences
    """
    input_seq=[]
    target_seq=[]
    for i in range(len(data)):
        input_seq.append(data[i][:-1])
        target_seq.append(data[i][1:])
    return input_seq, target_seq

#We'll input ten sequences, each with t_steps data points
batch_size=10
t_steps=1000
data=generate_data(batch_size,t_steps)
input_seq, target_seq=input_and_target(data)



#turn into tensors
input_seq = torch.Tensor(input_seq)
target_seq = torch.Tensor(target_seq)


# torch.cuda.is_available() checks and returns a Boolean True if a GPU is available, else it'll return False
is_cuda = torch.cuda.is_available()

# If we have a GPU available, we'll set our device to GPU. We'll use this device variable later in our code.
if is_cuda:
    device = torch.device("cuda")
    #print("GPU is available")
else:
    device = torch.device("cpu")
    #print("GPU not available, CPU used")

class Model(nn.Module):
    def __init__(self, input_size, output_size, hidden_dim, n_layers):
        
        super(Model, self).__init__()

        # Defining some parameters
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        #Defining the layers
        # RNN Layer
        self.rnn = nn.RNN(input_size, hidden_dim, n_layers, batch_first=True)   
        # Fully connected layer
        self.fc = nn.Linear(hidden_dim, output_size)
    
    def forward(self, x):
        
        batch_size = x.size(0)

        #Initializing hidden state for first input using method defined below
        hidden = self.init_hidden(batch_size)

        # Passing in the input and hidden state into the model and obtaining outputs
        out, hidden = self.rnn(x, hidden)
        
        # Reshaping the outputs such that it can be fit into the fully connected layer
        #out = out.contiguous().view(-1, self.hidden_dim)
        out = self.fc(out)
        
        return out, hidden
    
    def init_hidden(self, batch_size):
        # This method generates the first hidden state of zeros which we'll use in the forward pass
        hidden = torch.zeros(self.n_layers, batch_size, self.hidden_dim).to(device)
         # We'll send the tensor holding the hidden state to the device we specified earlier as well
        return hidden

# Instantiate the model with hyperparameters
model = Model(input_size=3, output_size=3, hidden_dim=10, n_layers=3)
# We'll also set the model to the device that we defined earlier (default is CPU)
model = model.to(device)

# Define hyperparameters
n_epochs = 100
lr=0.02

# Define Loss, Optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)


# Training Run
input_seq = input_seq.to(device)
outputs=[]
losses=[]
for epoch in range(1, n_epochs + 1):  
    optimizer.zero_grad() # Clears existing gradients from previous epoch
    #input_seq = input_seq.to(device)
    output, hidden = model(input_seq)

    output = output.to(device)
    target_seq = target_seq.to(device)


    loss = criterion(output, target_seq)
    loss.backward() # Does backpropagation and calculates gradients
    optimizer.step() # Updates the weights accordingly
    losses += [loss.item()]
    print(loss.item())
    #adaptive learning rate: decrease learning rate if cost increases, increase if it decreases
    #note that we decrease by more than we increase
    if epoch >=2:
        if loss.item() >= losses[epoch-1]:
            lr = lr*(5/6)
        else:
                lr=lr*1.1
def predict(size):
    data=generate_data(1,1)
    IC=torch.Tensor(data)
    output_seq=[]
    output_seq.append(IC[0])
    for i in range(size):
        out,hidden=model(IC)
        IC=out
        output_seq.append(IC[0])
    return output_seq
out=predict(100)
ox=[x.detach().numpy()[0][0] for x in out]
print(ox)
plt.scatter(range(101),ox)
plt.show()
