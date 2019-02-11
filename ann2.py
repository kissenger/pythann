import numpy as np
import random
import matplotlib.pyplot as plt
import scipy.stats
import sys

# ===============================================================
# References
# ===============================================================
# http://neuralnetworksanddeeplearning.com/chap1.html
# http://neuralnetworksanddeeplearning.com/chap2.html
# https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/

# ===============================================================
# Setup
# ===============================================================
#N_HIDDEN_NODES       = 4



# ===============================================================
# functions
# ===============================================================
def sigmoid(z):
    return 1/(1+np.exp(-z))
   
def sigmoid_prime(z):
    return sigmoid(z)*(1-sigmoid(z))


class Network(object):

	def __init__(self, sizes):
	# sizes is a list containing number of neurons per layer
	# eg [2,5,4] would be a 3 layer network with 2 input noted, 5 hidden nodes and 4 output
	# weights are initialised with mean 0 variance 1
		#np.random.seed(1)    #only used when repeatability is needed
		self.num_layers = len(sizes)
		self.sizes = sizes
		
		self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
		self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]
	
	def output(self, a):
	# runs a set of inputs through the entire network, with current set of weights
	# a is np.array of training inputs of dimensions [ni, 1], eg [[0],[1],[0]]
		for b, w in zip(self.biases, self.weights):
			a = sigmoid(np.dot(w, a)+b)
		return a
		
	def train(self, inputs, y, eta, err, epochs):
	# feeds training data through the network
	# a is np.array of training inputs of dimensions [ni, n_train]
	# y is np.array of training outputs of dimensions [no, n_train]
	# where ni is number of inputs, no is number of outputs, n_train is number of training rows
	# eta is the learning rate
	# err is the threshold error (break if below)
	# epochs is the max number of epochs to run
	
		cost = []
		for epoch in range(epochs):
	
			z = []
			a = [inputs]
			#print("a\n", a, "z\n", z)
			dC_by_db = [np.zeros(b.shape) for b in self.biases]
			dC_by_dw = [np.zeros(w.shape) for w in self.weights]
				
			# ===
			# forward propogation - pass input data through all layers of the network
			# ===
			for b, w in zip(self.biases, self.weights):
				z.append(np.dot(w, a[-1]) + b)   # z = weighted layer inputs (ie pre-activation)
				a.append(sigmoid(z[-1]))		 # a = activated layer outputs
				#print("biases\n", b, "\nweights\n", w, "\nz\n", z, "\na\n", a)
			
			C = np.array(0.5*np.square(a[-1] - y)).sum()   # C = total Cost
			if (C < err):
				break
			#print("C\n", C)
			
			# ===
			# backward propogation
			# ===

			d = (a[-1] - y)	* sigmoid_prime(z[-1])   # d = dC/da = dC/dz*dz/da
			dC_by_dw = np.dot(d, a[-2].transpose())  # da/dw = a[-2] so dC/dw = d*a[-2]
			self.weights[-1] -= eta * dC_by_dw
			
			for layer in range(self.num_layers-3,-1,-1):
				d = np.dot(self.weights[layer+1].transpose(), d) * sigmoid_prime(z[layer])
				dC_by_dw = np.dot(d, a[layer].transpose())	
				self.weights[layer] += -eta * dC_by_dw
				cost.append(C)
		
		if (epoch == epochs-1):
			print('network did not converge')
		else:
			print('network converged in ', epoch, ' epochs\nfinal weights:\n', self.weights)
		plt.plot(range(0,len(cost),1), cost)
		plt.show()
	
