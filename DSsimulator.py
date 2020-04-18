from __future__ import unicode_literals, print_function, division
from io import open
import glob
import os
import numpy as np
from scipy.integrate import odeint
import torch
import torch.nn as nn

def DE(s,t):
    """
    defines a system of linear odes that we will integrate in generatesamp()
    """
    x=s[0]
    y=s[1]
    dxdt = -y
    dydt= x
    return [dxdt, dydt]

def generatesamp(s0, t_steps):
    """
    returns an array with x and y values for every timestep
    """
    t=np.linspace(0,t_steps)
    s=odeint(DE,s0,t)
    return s

import random

def generate_data(batch_size,t_steps):
    """
    returns a batch_size x t_steps matrix
    """
    #batch_size number of sequences, each w t_steps data points
    data = np.zeros((batch_size, t_steps,1), dtype =np.float32)
    for i in range(batch_size):
        s0 = [random.uniform(-1,1),random.uniform(-1,1)]
        s=generatesamp(s0, t_steps)
        for j in range(t_steps):
            #data populated with x(t) for each t
            data[i][j]=s[j][1]
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

#We'll input ten sequences, each with ten data points
batch_size=10
t_steps=10
data=generate_data(batch_size,t_steps)
input_seq, target_seq=input_and_target(data)
#print("input: ", input_seq, "target: ", target_seq)
input_seq=np.array(input_seq)
#print("Input shape: {} --> (Batch Size, Sequence Length, One-Hot Encoding Size)".format(input_seq.shape))


#turn into tensors
input_seq = torch.from_numpy(input_seq)
target_seq = torch.Tensor(target_seq)
print("input: ", len(input_seq[0]), "target: ", len(target_seq[0]))
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
        out = out.contiguous().view(-1, self.hidden_dim)
        out = self.fc(out)
        
        return out, hidden
    
    def init_hidden(self, batch_size):
        # This method generates the first hidden state of zeros which we'll use in the forward pass
        hidden = torch.zeros(self.n_layers, batch_size, self.hidden_dim).to(device)
         # We'll send the tensor holding the hidden state to the device we specified earlier as well
        return hidden

# Instantiate the model with hyperparameters
model = Model(input_size=1, output_size=9, hidden_dim=10, n_layers=1)
# We'll also set the model to the device that we defined earlier (default is CPU)
model = model.to(device)

# Define hyperparameters
n_epochs = 100
lr=0.01

# Define Loss, Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)


# Training Run
input_seq = input_seq.to(device)
for epoch in range(1, n_epochs + 1):
    optimizer.zero_grad() # Clears existing gradients from previous epoch
    #input_seq = input_seq.to(device)
    output, hidden = model(input_seq)

    output = output.to(device)
    target_seq = target_seq.to(device)
    #getting the error index error: target -1 out of bounds
    #print("output:", output, "target:", target_seq)
    loss = criterion(output, target_seq.view(-1).long())
    loss.backward() # Does backpropagation and calculates gradients
    optimizer.step() # Updates the weights accordingly
    
    if epoch%10 == 0:
        print('Epoch: {}/{}.............'.format(epoch, n_epochs), end=' ')
        print("Loss: {:.4f}".format(loss.item()))
        print("output:", output[0], "target:", target_seq[0])
 

 
