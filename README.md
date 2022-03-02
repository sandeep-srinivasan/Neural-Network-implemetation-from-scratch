# Neural-Network-implemetation-from-scratch
Neural network implementation using numpy, sklearn, matplotlib etc

# 1. Multilayer Perceptron

Implemented a two-layer perceptron with the backpropagation algorithm to solve the parity problem. The desired output for the parity problem is 1 if an input pattern contains an odd number of 1's and -1 otherwise.

How it works:
The Network consists with 5 binary (+1/-1) input elements, 5 hidden units for the first layer, and one output unit for the second layer. The learning procedure is stopped when     an absolute error (difference) of 0.05 is reached for every input pattern. Other implementation details are:

● Initialized all weights and biases to random numbers between -1 and 1.

● Used tanh with a=1 (i.e.,) as the activation function for all units.

● Included a momentum term in the weight update with, α=0.8 and report its effect on the speed of training for each value of η.

● Replaced the tanh function with another nonlinearity which is leaky relu.

![image](https://user-images.githubusercontent.com/42225976/156085331-3ec1daa9-4a57-4a9f-a6ac-0339380b879c.png)

# 2. Radial Basis Functions (RBF)

Implemented an RBF network for one input variable, one output variable and Gaussian basis functions. 

How it works:

● Generated a set of 75 data points by sampling the function, h(x) = 0.5 + 0.4sin(3πx) with added uniform noise in the interval [-0.1, 0.1] and with x values taken randomly from a uniform distribution in the interval [0.0, 1.0]. 

● Determined Gaussian centers by your implementation of the K-means algorithm, and set the variance of each cluster according to the variance of the cluster. If a cluster contains only one sample point, used the mean variance of all the other clusters as its variance.

● Used the Least Mean Square rule for weight update. 

● Varied the number of bases in the range of 3, 6, 8, 12, and 16.

![image](https://user-images.githubusercontent.com/42225976/156085883-64ec47c2-adcf-4bcd-b5ee-bde3bca87289.png)

![image](https://user-images.githubusercontent.com/42225976/156085741-145bc97e-48f9-4d6f-ab7f-2da4c7f318f1.png)

# 3. Restricted Boltzmann Machine

Developed a Restricted Boltzmann Machine to learn a collaborative filter for predicting ice cream preferences.
The dataset consists of 120 different examples for 10 icecream kinds.

How it works:

![image](https://user-images.githubusercontent.com/42225976/156087139-9ac672dc-be99-4d1d-a226-fc34c88ca959.png)



