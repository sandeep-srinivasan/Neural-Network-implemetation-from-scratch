# Neural-Network-implemetation-from-scratch
Neural network implementation using numpy, sklearn, matplotlib etc

1. Implemented a two-layer perceptron with the backpropagation algorithm to solve the parity problem. The desired output for the parity problem is 1 if an input pattern contains an odd number of 1's and -1 otherwise.

How it works:
The Network consists with 5 binary (+1/-1) input elements, 5 hidden units for the first layer, and one output unit for the second layer. The learning procedure is stopped when an
absolute error (difference) of 0.05 is reached for every input pattern. Other implementation details are:
● Initialized all weights and biases to random numbers between -1 and 1.
● Used tanh with a=1 (i.e.,) as the activation function for all units.
● Included a momentum term in the weight update with, U+03B1=0.8 and report its effect on the speed of training for each value of U+03B1.
● Replaced the tanh function with another nonlinearity which is leaky relu.
