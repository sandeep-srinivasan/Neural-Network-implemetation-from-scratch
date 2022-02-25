#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt


# In[2]:


#K-means Class
class k_means():
    def __init__(self, x, max_epoch, number_of_bases, variance):
    #where x=input array, max_epoch=maximum number of epochs=100,number of bases=number of neurons in the hidden layer, variance = type of gaussian width of clusters i.e., 'different' or 'same'
    
        self.x = x
        self.max_epoch = max_epoch
        self.k = number_of_bases
        self.variance = variance
        
    def gaussian_widths(self, cluster_center, cluster_labels):
        if(self.variance == 'different'):
            sigmas = np.zeros((self.k,), dtype = float)
            for j in range(self.k):
                sigmas[j] = np.sqrt(np.mean((cluster_center[j] - self.x[cluster_labels == j])**2))
                
            #If a cluster contains only one sample point, use as its variance the mean variance of all the other clusters.
            for l in range(self.k):
                if(sigmas[l] == 0):
                    sigmas[l] = np.mean([x for i,x in enumerate(sigmas) if i!=l])
            return(sigmas)
        
        elif(self.variance == 'same'):
            #standard deviation = Maximum distance between centres or two hidden neurons/sqrt(2*number of clusters)
            sigma = (np.max(cluster_center) - np.min(cluster_center))/np.sqrt(2*self.k)
            sigmas = np.tile(sigma, self.k)
            return(sigmas)
        
    def nearest_cluster_center(self, sample, cluster_centers):
        distance_from_centre = [np.sqrt(np.sum((sample-centroid)**2)) for centroid in cluster_centers]
        nearest_idx = np.argmin(distance_from_centre)
        return(nearest_idx)
    
    def new_clusters(self,cluster_centers):
        clusters = [[] for i in range(self.k)]
        for idx, sample in enumerate(self.x):
            centroid_indices = self.nearest_cluster_center(sample, cluster_centers)
            clusters[centroid_indices].append(idx)
        return(clusters)
    
    def new_cluster_centers(self,clusters):
        cluster_centers = np.zeros((self.k,))
        for cluster_idx, cluster_sample in enumerate(clusters):
            cluster_mean = np.mean(self.x[cluster_sample], axis=0)
            cluster_centers[cluster_idx] = cluster_mean
        return(cluster_centers)
    
    def new_cluster_labels(self,clusters):
        labels = np.empty((len(self.x),))
        for cluster_idx,cluster_sample in enumerate(clusters):
            for cluster_sample_idx in cluster_sample:
                labels[cluster_sample_idx] = cluster_idx
        return(labels)
        
    def clustering_output(self,x):
        #Intialize the cluster_centers
        cluster_centers = np.random.choice(self.x, size = self.k, replace=False)
        
        #Optimize the cluster_centers
        for i in range(self.max_epoch):
            #Update the clusters
            clusters = self.new_clusters(cluster_centers)
            
            #Update the cluster_centers
            previous_cluster_centers = cluster_centers
            cluster_centers = self.new_cluster_centers(clusters)
            
            if([np.sqrt(np.sum((previous_cluster_centers[i]-cluster_centers[i])**2)) for i in range(self.k)]) == 0:
                break
        
        cluster_labels = self.new_cluster_labels(clusters)
        
        sigmas = self.gaussian_widths(cluster_centers,cluster_labels)
        
        return(cluster_centers, sigmas)


# In[3]:


#RBF network Class
class rbf_network():
    def __init__(self, x, d, eta, number_of_bases, variance, cluster_centers, sigmas, max_epoch):
        self.x = x
        self.d = d
        self.eta = eta
        self.number_of_bases = number_of_bases
        self.variance = variance
        self.cluster_centers = cluster_centers
        self.sigmas = sigmas
        self.max_epoch = max_epoch
       
        #Intialize the weights and a bias term
        self.weights = np.random.uniform(-1, 1, (number_of_bases, ))
        self.bias = 1.0
        self.bias_weight = np.random.uniform(-1, 1, (1, ))
        kMeans = k_means(x,max_epoch,number_of_bases,variance)
        self.cluster_centers, self.sigmas = kMeans.clustering_output(x)
        
    def rbf_gaussian(self, x, cluster_centers, sigmas):
        return(np.exp((-(x-cluster_centers)**2)/(2*sigmas**2)))
        
    def weight_update(self, x, d, cluster_centers, sigmas, eta):
        #LMS rule to update the weights
        updated_weights = []
        
        for n in range(self.max_epoch):
            actual_output = np.zeros((len(self.x),))
            for i in range(len(self.x)):
                actual_output[i] = np.dot(self.rbf_gaussian(x[i],cluster_centers,sigmas).T, self.weights) + np.dot(self.bias, self.bias_weight)
                
                error = d[i] - actual_output[i]
                self.weights = self.weights + (self.eta*error*self.rbf_gaussian(x[i],cluster_centers,sigmas))
                
                self.bias_weight = self.bias_weight + self.eta*error
            
        updated_weights.append(self.weights)
        updated_weights.append(self.bias_weight)
            
        actual_output = np.asarray(actual_output)
        updated_weights = np.asarray(updated_weights)
            
        return(actual_output, updated_weights)


# In[4]:


#X-values
x = np.sort(np.random.uniform(0,1,(75,)))

#Noise
noise = np.random.uniform(-0.1,0.1, (x.shape))

#Original function h(x)
h = (0.5 + 0.4*np.sin(3*np.pi*x))

#Desired output of data points
d = (0.5 + 0.4*np.sin(3*np.pi*x)) + noise

#Maximum number of epochs
max_epoch = 100


# In[5]:


#Graph plot of each case of combinations of variance, learning rates and number of bases
for variance in ('different', 'same'):
    for eta in (0.01, 0.02):
        for number_of_bases in (3,6,8,12,16):
    
            kMeans = k_means(x,max_epoch,number_of_bases,variance)
            cluster_centers, sigmas = kMeans.clustering_output(x)

            RBF = rbf_network(x, d, eta, number_of_bases, variance, cluster_centers, sigmas, max_epoch)

            actual_output, updated_weights = RBF.weight_update(x, d, cluster_centers, sigmas, eta)
    
            plt.figure(figsize=(10,10))

            plt.xlabel('X-values', fontsize = 15)
            plt.ylabel('Function of X', fontsize = 15)
            plt.scatter(x,d, color='red', s=30, label="Data points")
            plt.plot(x, h, color='darkblue', lw=4, label="Original function")
            plt.plot(x,actual_output, color='cyan', lw=4, label="RBF network")
            plt.xlim(-0.1,1.1)
            plt.ylim(-0.1,1.1)
            plt.title("For number of bases = " + str(number_of_bases) + ", eta = " + str(eta) + ", Variance of clusters = " + str(variance))
            plt.legend(loc=9)
            plt.savefig('v'+str(variance)+'_e'+str(eta)+'_k'+str(number_of_bases)+'.png')
plt.show()

