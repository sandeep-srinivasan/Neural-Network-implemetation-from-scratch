#!/usr/bin/env python
# coding: utf-8

# # Imports

# In[1]:


import numpy as np
import itertools


# In[2]:


#Input Pattern 
X = np.insert(np.asarray(list(itertools.product([-1, 1], repeat=5))), 0, values=1, axis=1)


# In[3]:


#Desired Outputs
D = (np.sum((X[:,1:]==1), axis=1) % 2 == 1)*1
D[D==0] = -1


# In[4]:


#Network
class twoLayerPerceptron(object):
    def __init__(self, eta, alpha, activation_function, threshold):
        #Initilaizing the Network
        self.inputSize = 5
        self.hiddenSize = 8
        self.outputSize = 1
        
        self.activation_function = activation_function
        
        #Initializing the weights
        self.weights_hidden = np.random.uniform(-1, 1, (self.inputSize+1, self.hiddenSize))
        self.weights_output = np.random.uniform(-1, 1, (self.hiddenSize, self.outputSize))
        
        #For alpha term
        self.delta_hidden = np.zeros(((self.inputSize+1, self.hiddenSize)))
        self.delta_output = np.zeros((self.hiddenSize, self.outputSize))
        
    def activation(self, x):
        if self.activation_function == 'tanh':
            return(np.tanh(x))
        elif self.activation_function == 'leaky_relu':
            return(np.where(x > 0, x, x * 0.01))
    
    #FeedForward function
    def FP(self, x):
        self.output_hidden = self.activation(np.dot(x,self.weights_hidden).T)
        self.output = self.activation(np.dot(self.weights_output.T, self.output_hidden))
        return(self.output_hidden, self.output)
    
    #Function for derivative of tanh
    def derivative_activation(self, a):
        if self.activation_function == 'tanh':
            return(1-a**2)
        elif self.activation_function == 'leaky_relu':
            d = np.ones_like(a)
            d[a < 0] = 0.01
            return(d)
    
    #Backpropogation function
    def BP(self, x, d, y1, y, eta, alpha):
        #Change in error at output
        self.output_error = d - y
        self.output_delta = np.multiply(self.derivative_activation(y), self.output_error)
        
        self.delta_output = eta*np.dot(y1, self.output_delta) + (alpha*self.delta_output)
        
        #Change in error at output because of the hidden layers
        self.hidden_output_error = np.dot(self.weights_output, self.output_delta)
        self.hidden_delta = np.multiply(self.derivative_activation(y1), self.hidden_output_error)
        
        self.delta_hidden = eta*np.dot(self.hidden_delta, x).T + (alpha*self.delta_hidden)
        
        #Updating the weights
        self.weights_hidden += self.delta_hidden
        self.weights_output += self.delta_output
        
    #Training the Network
    def train(self, X, D,eta,alpha,threshold):
        epoch = 0
        total_error = True
        
        while (total_error):
            errors = []
            for i in range(32):
                x = X[i].reshape(1,-1)
                d = D[i].reshape(1,-1)
                y1, y = self.FP(x)
                self.BP(x,d,y1,y,eta,alpha)
                
                if(np.abs(d - y) > threshold):
                    absolute_error = True
                else:
                    absolute_error = False
                    
                errors.append(absolute_error)
            
            epoch += 1
            
            if(all(item == False for item in errors)):
                total_error = False
            else:
                total_error = True
                
        print("The number of epochs to reach the stopping criterion for eta =",np.round(eta,3),", alpha =", alpha, ":", epoch)

for eta in np.arange(0.005, 0.051, 0.005):
    alpha = 0.0
    activation_function = 'tanh'
    threshold = 0.1
    MLP = twoLayerPerceptron(eta,alpha,activation_function,threshold)
    MLP.train(X,D,eta,alpha,threshold)
    print("The actual output is:\n", MLP.FP(X)[1])  

for eta in np.arange(0.005, 0.051, 0.005):
    alpha = 0.8
    activation_function = 'tanh'
    MLP = twoLayerPerceptron(eta,alpha,activation_function,threshold)
    MLP.train(X,D,eta,alpha,threshold)
    print("The actual output is:\n", MLP.FP(X)[1])


# # Bonus

# In[5]:


for eta in np.arange(0.05, 0.51, 0.05):
    alpha = 0.0
    activation_function = 'leaky_relu'
    threshold = 0.99
    MLP = twoLayerPerceptron(eta,alpha,activation_function,threshold)
    MLP.train(X,D,eta,alpha,threshold)
    print("The actual output is:\n", MLP.FP(X)[1])

