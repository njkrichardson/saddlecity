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

def sample(s0, t_steps):
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
    # batch_size number of sequences, each with t_steps data points
    data = np.zeros((batch_size, t_steps,3), dtype =np.float32)
    for i in range(batch_size):
        s0 = [random.uniform(-1,1),random.uniform(-1,1),random.uniform(-1,1)]
        s = sample(s0, t_steps)
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