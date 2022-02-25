#!/usr/bin/env python
# coding: utf-8

# # Part 1

# In[1]:


import numpy as np
from sklearn.model_selection import train_test_split


class RBM():
    def __init__(self, data, num_hidden, eta, max_epochs, cd_iterations, bias):
        
        self.num_visible = data.shape[1]
        self.num_hidden = num_hidden
        self.num_examples = data.shape[0]
        self.eta = eta
        self.max_epochs = max_epochs
        self.cd_iterations = cd_iterations
        self.bias = bias
        #num_visible = number of visible units
        #num_hidden = number of hidden units
        #num_examples = number of examples/observations in the dataset
        #eta = learning rate
        #max_epochs = number of epochs during training
        #cd_iterations = number of iterations in Contrastive divergence phase
        #Bias = A boolean of yes/no
        
        self.weights = np.random.uniform(0,1,(self.num_visible, self.num_hidden))
        #Weights = Weights of the RBM model
        
        if(self.bias == 'yes'):
            # Insert weights for the bias units into the first row and first column.
            self.weights = np.insert(self.weights, 0, 0, axis = 0)
            self.weights = np.insert(self.weights, 0, 0, axis = 1)
        
    def sigmoid(self, z):
        return (1/(1+np.exp(-z)))
        
    def train(self, data):
        #Contrastive Divergence Training phase
        
        if(self.bias == 'yes'):
            # Insert bias units of 1 into the first column.
            data = np.insert(data, 0, 1, axis = 1)
        
        batch_size = 12
        batches = int(self.num_examples/batch_size)
        
        for i in range(self.max_epochs):
            for batch in np.arange(batches):
                x = data[batch*batch_size:(batch*batch_size) + batch_size,:]
                
                h0_activation = np.dot(x, self.weights)   
                h0_probability = self.sigmoid(h0_activation) 
                
                if(self.bias == 'yes'):
                    h0_probability[:,0] = 1 # Fix the bias unit
                    h0_states = (h0_probability > np.random.uniform(0,1,(batch_size, self.num_hidden+1)))*1.0
                    
                    v0h0_associations = np.dot(x.T, h0_probability) #v0*h0 of shape 11x5
            
                    v1_activation = np.dot(h0_states, self.weights.T)      
                    v1_probability = self.sigmoid(v1_activation) 
                    
                    v1_probability[:,0] = 1 # Fix the bias unit
                    
                    h1_activation = np.dot(v1_probability, self.weights) 
                    h1_probability = self.sigmoid(h1_activation) 
            
                    v1h1_associations = np.dot(v1_probability.T, h1_probability) #v1*h1 of shape 11x5
            
                    h1_states = (h1_probability > np.random.uniform(0,1,(batch_size, self.num_hidden+1)))*1.0
                    v2_activation = np.dot(h1_states, self.weights.T)      
                    v2_probability = self.sigmoid(v2_activation) 
            
                    h2_activation = np.dot(v2_probability, self.weights) 
                    h2_probability = self.sigmoid(h2_activation) 
            
                    v2h2_associations = np.dot(v2_probability.T, h2_probability) #v2*h2 of shape 11x5
            
                    h2_states = (h2_probability > np.random.uniform(0,1,(batch_size, self.num_hidden+1)))*1.0
                    v3_activation = np.dot(h2_states, self.weights.T)      
                    v3_probability = self.sigmoid(v3_activation) 
            
                    h3_activation = np.dot(v3_probability, self.weights) 
                    h3_probability = self.sigmoid(h3_activation) 
            
                    v3h3_associations = np.dot(v3_probability.T, h3_probability) #v3*h3 of shape 11x5
                
                else:
                    h0_states = (h0_probability > np.random.uniform(0,1,(batch_size, self.num_hidden)))*1.0
                
                    v0h0_associations = np.dot(x.T, h0_probability) #v0*h0 of shape 10x4
            
                    v1_activation = np.dot(h0_states, self.weights.T)      
                    v1_probability = self.sigmoid(v1_activation) 
            
                    h1_activation = np.dot(v1_probability, self.weights) 
                    h1_probability = self.sigmoid(h1_activation) 
            
                    v1h1_associations = np.dot(v1_probability.T, h1_probability) #v1*h1 of shape 10x4
            
                    h1_states = (h1_probability > np.random.uniform(0,1,(batch_size, self.num_hidden)))*1.0
                    v2_activation = np.dot(h1_states, self.weights.T)      
                    v2_probability = self.sigmoid(v2_activation) 
            
                    h2_activation = np.dot(v2_probability, self.weights) 
                    h2_probability = self.sigmoid(h2_activation) 
            
                    v2h2_associations = np.dot(v2_probability.T, h2_probability) #v2*h2 of shape 10x4
            
                    h2_states = (h2_probability > np.random.uniform(0,1,(batch_size, self.num_hidden)))*1.0
                    v3_activation = np.dot(h2_states, self.weights.T)      
                    v3_probability = self.sigmoid(v3_activation) 
            
                    h3_activation = np.dot(v3_probability, self.weights) 
                    h3_probability = self.sigmoid(h3_activation) 
            
                    v3h3_associations = np.dot(v3_probability.T, h3_probability) #v3*h3 of shape 10x4
            
            if(self.cd_iterations == 1):
                #Weight update
                self.weights += np.multiply(self.eta, (v0h0_associations - v1h1_associations)/batch_size)
            
                mae = np.mean((np.abs(x - v1_probability))) #absolute difference of v0 and v1
                
            elif(self.cd_iterations == 3):
                #Weight update
                self.weights += np.multiply(self.eta, (v0h0_associations - v3h3_associations)/batch_size)
            
                mae = np.mean((np.abs(x - v3_probability))) #absolute difference of v0 and v1
                
        print("For eta:", np.round(eta,2), "and epoch:", epoch, "the Mean Absolute Error (MAE) is:", np.round(mae,4))
                
        if(self.cd_iterations == 1 and self.bias == 'yes'):
            reconstructed = (v1_probability > np.random.uniform(0,1,(batch_size, self.num_visible+1)))*1.0
            reconstructed = reconstructed[:,1:]
        elif(self.cd_iterations == 1 and self.bias == 'no'):
            reconstructed = (v1_probability > np.random.uniform(0,1,(batch_size, self.num_visible)))*1.0
        elif(self.cd_iterations == 3 and self.bias == 'yes'):
            reconstructed = (v3_probability > np.random.uniform(0,1,(batch_size, self.num_visible+1)))*1.0
            reconstructed = reconstructed[:,1:]
        elif(self.cd_iterations == 3 and self.bias == 'no'):
            reconstructed = (v3_probability > np.random.uniform(0,1,(batch_size, self.num_visible)))*1.0
            
        return(reconstructed)


# In[2]:


#Read the icecream data
ice_cream = np.loadtxt('icecream.csv',dtype = 'int', delimiter=',')


# In[3]:


#Baseline model with no bias, for 4 hidden units and for a single iteration of Contrastive Divergence
if __name__ == '__main__':
    num_hidden = 4
    cd_iterations = 1
    bias = 'no'
    for eta in np.arange(0.05, 0.26, 0.05):
        for epoch in np.arange(1000,5001,1000):
            rbm = RBM(ice_cream, num_hidden, eta, epoch, cd_iterations, bias)
            reconstructed = rbm.train(ice_cream)


# In[4]:


print(reconstructed)
print(ice_cream[108:])


# # Part 2

# In[5]:


#Model with bias, for 4 hidden units and for a single iteration of Contrastive Divergence
if __name__ == '__main__':
    num_hidden = 4
    cd_iterations = 1
    bias = 'yes'
    for eta in np.arange(0.05, 0.26, 0.05):
        for epoch in np.arange(1000,5001,1000):
            rbm = RBM(ice_cream, num_hidden, eta, epoch, cd_iterations, bias)
            reconstructed = rbm.train(ice_cream)


# In[6]:


print(reconstructed)
print(ice_cream[108:])


# In[7]:


#Model with no bias, for 8 hidden units and for a single iteration of Contrastive Divergence
if __name__ == '__main__':
    num_hidden = 8
    cd_iterations = 1
    bias = 'no'
    for eta in np.arange(0.05, 0.26, 0.05):
        for epoch in np.arange(1000,5001,1000):
            rbm = RBM(ice_cream, num_hidden, eta, epoch, cd_iterations, bias)
            reconstructed = rbm.train(ice_cream)


# In[8]:


print(reconstructed)
print(ice_cream[108:])


# In[9]:


#Model with bias, for 8 hidden units and for a single iteration of Contrastive Divergence
if __name__ == '__main__':
    num_hidden = 8
    cd_iterations = 1
    bias = 'yes'
    for eta in np.arange(0.05, 0.26, 0.05):
        for epoch in np.arange(1000,5001,1000):
            rbm = RBM(ice_cream, num_hidden, eta, epoch, cd_iterations, bias)
            reconstructed = rbm.train(ice_cream)


# In[10]:


print(reconstructed)
print(ice_cream[108:])


# In[11]:


#Model with no bias, for 4 hidden units and for three iterations of Contrastive Divergence
if __name__ == '__main__':
    num_hidden = 4
    cd_iterations = 3
    bias = 'no'
    for eta in np.arange(0.05, 0.26, 0.05):
        for epoch in np.arange(1000,5001,1000):
            rbm = RBM(ice_cream, num_hidden, eta, epoch, cd_iterations, bias)
            reconstructed = rbm.train(ice_cream)


# In[12]:


print(reconstructed)
print(ice_cream[108:])


# # Bonus 1

# In[13]:


class RBM_revised():
    def __init__(self, data, num_hidden, eta, max_epochs, cd_iterations, bias, batch_size, bonus):
        
        self.num_visible = data.shape[1]
        self.num_hidden = num_hidden
        self.num_examples = data.shape[0]
        self.eta = eta
        self.max_epochs = max_epochs
        self.cd_iterations = cd_iterations
        self.bias = bias
        self.data = data
        self.batch_size = batch_size
        self.bonus = bonus
        #num_visible = number of visible units
        #num_hidden = number of hidden units
        #num_examples = number of examples/observations in the dataset
        #eta = learning rate
        #max_epochs = number of epochs during training
        #cd_iterations = number of iterations in Contrastive divergence phase
        #Bias = A boolean of yes/no
        
        self.train_data, self.test_data = train_test_split(self.data, test_size=0.1, shuffle = False)
        
        self.weights = np.random.uniform(0,1,(self.num_visible, self.num_hidden))
        #Weights = Weights of the RBM model
        
        if(self.bias == 'yes'):
            # Insert weights for the bias units into the first row and first column.
            self.weights = np.insert(self.weights, 0, 0, axis = 0)
            self.weights = np.insert(self.weights, 0, 0, axis = 1)
        
    def sigmoid(self, z):
        return (1/(1+np.exp(-z)))
        
    def train(self):
        #Contrastive Divergence Training phase
        
        if(self.bias == 'yes'):
            # Insert bias units of 1 into the first column.
            self.train_data = np.insert(self.train_data, 0, 1, axis = 1)
        
        batches = int(len(self.train_data)/batch_size)
        
        for i in range(self.max_epochs):
            for batch in np.arange(batches):
                x = self.train_data[batch*self.batch_size:(batch*self.batch_size) + self.batch_size,:]
                
                h0_activation = np.dot(x, self.weights)   
                h0_probability = self.sigmoid(h0_activation) 
                
                if(self.bias == 'yes'):
                    h0_probability[:,0] = 1 # Fix the bias unit
                    h0_states = (h0_probability > np.random.uniform(0,1,(batch_size, self.num_hidden+1)))*1.0
                    
                    v0h0_associations = np.dot(x.T, h0_probability) #v0*h0 of shape 11x5
            
                    v1_activation = np.dot(h0_states, self.weights.T)      
                    v1_probability = self.sigmoid(v1_activation) 
                    
                    v1_probability[:,0] = 1 # Fix the bias unit
                    
                    h1_activation = np.dot(v1_probability, self.weights) 
                    h1_probability = self.sigmoid(h1_activation) 
            
                    v1h1_associations = np.dot(v1_probability.T, h1_probability) #v1*h1 of shape 11x5
            
                    h1_states = (h1_probability > np.random.uniform(0,1,(batch_size, self.num_hidden+1)))*1.0
                    v2_activation = np.dot(h1_states, self.weights.T)      
                    v2_probability = self.sigmoid(v2_activation) 
            
                    h2_activation = np.dot(v2_probability, self.weights) 
                    h2_probability = self.sigmoid(h2_activation) 
            
                    v2h2_associations = np.dot(v2_probability.T, h2_probability) #v2*h2 of shape 11x5
            
                    h2_states = (h2_probability > np.random.uniform(0,1,(batch_size, self.num_hidden+1)))*1.0
                    v3_activation = np.dot(h2_states, self.weights.T)      
                    v3_probability = self.sigmoid(v3_activation) 
            
                    h3_activation = np.dot(v3_probability, self.weights) 
                    h3_probability = self.sigmoid(h3_activation) 
            
                    v3h3_associations = np.dot(v3_probability.T, h3_probability) #v3*h3 of shape 11x5
                
                else:
                    h0_states = (h0_probability > np.random.uniform(0,1,(batch_size, self.num_hidden)))*1.0
                
                    v0h0_associations = np.dot(x.T, h0_probability) #v0*h0 of shape 10x4
            
                    v1_activation = np.dot(h0_states, self.weights.T)      
                    v1_probability = self.sigmoid(v1_activation) 
            
                    h1_activation = np.dot(v1_probability, self.weights) 
                    h1_probability = self.sigmoid(h1_activation) 
            
                    v1h1_associations = np.dot(v1_probability.T, h1_probability) #v1*h1 of shape 10x4
            
                    h1_states = (h1_probability > np.random.uniform(0,1,(batch_size, self.num_hidden)))*1.0
                    v2_activation = np.dot(h1_states, self.weights.T)      
                    v2_probability = self.sigmoid(v2_activation) 
            
                    h2_activation = np.dot(v2_probability, self.weights) 
                    h2_probability = self.sigmoid(h2_activation) 
            
                    v2h2_associations = np.dot(v2_probability.T, h2_probability) #v2*h2 of shape 10x4
            
                    h2_states = (h2_probability > np.random.uniform(0,1,(batch_size, self.num_hidden)))*1.0
                    v3_activation = np.dot(h2_states, self.weights.T)      
                    v3_probability = self.sigmoid(v3_activation) 
            
                    h3_activation = np.dot(v3_probability, self.weights) 
                    h3_probability = self.sigmoid(h3_activation) 
            
                    v3h3_associations = np.dot(v3_probability.T, h3_probability) #v3*h3 of shape 10x4
            
            if(self.cd_iterations == 1):
                #Weight update
                self.weights += np.multiply(self.eta, (v0h0_associations - v1h1_associations)/batch_size)
            
                mae = np.mean((np.abs(x - v1_probability))) #absolute difference of v0 and v1
                
            elif(self.cd_iterations == 3):
                #Weight update
                self.weights += np.multiply(self.eta, (v0h0_associations - v3h3_associations)/batch_size)
            
                mae = np.mean((np.abs(x - v3_probability))) #absolute difference of v0 and v1
                
        print("For eta:", np.round(eta,2), "and epoch:", epoch, "the Mean Absolute Error (MAE) is:", np.round(mae,4))
                
        if(self.cd_iterations == 1 and self.bias == 'yes'):
            reconstructed = (v1_probability > np.random.uniform(0,1,(batch_size, self.num_visible+1)))*1.0
            reconstructed = reconstructed[:,1:]
        elif(self.cd_iterations == 1 and self.bias == 'no'):
            reconstructed = (v1_probability > np.random.uniform(0,1,(batch_size, self.num_visible)))*1.0
        elif(self.cd_iterations == 3 and self.bias == 'yes'):
            reconstructed = (v3_probability > np.random.uniform(0,1,(batch_size, self.num_visible+1)))*1.0
            reconstructed = reconstructed[:,1:]
        elif(self.cd_iterations == 3 and self.bias == 'no'):
            reconstructed = (v3_probability > np.random.uniform(0,1,(batch_size, self.num_visible)))*1.0
            
    def test(self):
        if(self.bonus == 1):
        
            indices = []
        
            for i in range(self.test_data.shape[0]):
                arr, idx = np.unique(self.test_data[i], return_index=True)
                self.test_data[i][idx] = -1
                indices.append(idx)
    
        if(self.bias == 'yes'):
            # Insert bias units of 1 into the first column.
            self.test_data = np.insert(self.test_data, 0, 1, axis = 1)
            
        hidden_activation = np.dot(self.test_data, self.weights)
        hidden_probability = self.sigmoid(hidden_activation)
        
        visible_activation = np.dot(hidden_probability, self.weights.T)
        visible_probability = self.sigmoid(visible_activation)
        
        if(self.bias=='yes'):
            visible_probability = visible_probability[:,1:]
        
        if(self.bonus == 1):
            count = 0
            for i in range(visible_probability.shape[0]):
                if(visible_probability[i][indices[i][1]] > visible_probability[i][indices[i][0]]):
                    count += 1
            
            return(visible_probability, count)
        else:
            return(visible_probability)


# In[14]:


#Model with bias, for 8 hidden units and for three iterations of Contrastive Divergence
for i in range(10):
    if __name__ == '__main__':
        num_hidden = 8
        cd_iterations = 3
        bias = 'yes'
        eta = 0.25
        epoch = 5000
        batch_size = int(np.ceil(ice_cream.shape[0]/10))
        rbm_revised = RBM_revised(ice_cream, num_hidden, eta, epoch, cd_iterations, bias, batch_size,1)
        rbm_revised.train()
        print("The original test array is\n:", ice_cream[108:])
        reconstructed, count = rbm_revised.test()
        print("The reconstructed test array is:\n", np.round(reconstructed,3))
        print("The number of times the positive instance had a higher probability than the negative instance is:", count)


# # Bonus 2

# In[15]:


#Read the Jester Simple data
jester_simple = np.loadtxt('jester-simple.csv',dtype = 'int', delimiter=',')


# In[16]:


#Model with bias, for different hidden units and for a single iteration of Contrastive Divergence
if __name__ == '__main__':
    cd_iterations = 1
    bias = 'yes'
    eta = 0.25
    epoch = 10
    batch_size = int(np.ceil(jester_simple.shape[0]/10))
    for num_hidden in np.arange(4,11,1):
        rbm_revised = RBM_revised(jester_simple, num_hidden, eta, epoch, cd_iterations, bias, batch_size,2)
        print("For " + str(num_hidden)+ " hidden units")
        rbm_revised.train()
        print("The original test array is\n:", jester_simple[len(jester_simple)-2499:])
        reconstructed = rbm_revised.test()
        print("The reconstructed test array is:\n", np.round(reconstructed,3))

