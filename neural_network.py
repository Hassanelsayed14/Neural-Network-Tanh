#!/usr/bin/env python
# coding: utf-8

# In[1]:


import random

def tanh(x):
    return (2 / (1 + 2.71828 ** (-2 * x))) - 1

def dot_product(inputs, weights):
    return sum(x * w for x, w in zip(inputs, weights))

def neural_network(inputs, weights1, weights2, b1, b2):
    hidden_layer = [tanh(dot_product(inputs, w) + b1) for w in weights1]
    
    output = tanh(dot_product(hidden_layer, weights2) + b2)
    
    return output

inputs = [0.5, 0.3, 0.2]

weights1 = [[random.uniform(-0.5, 0.5) for _ in range(3)] for _ in range(4)]
weights2 = [random.uniform(-0.5, 0.5) for _ in range(4)]

b1, b2 = 0.5, 0.7

output = neural_network(inputs, weights1, weights2, b1, b2)

print("Output of the network:", output)

