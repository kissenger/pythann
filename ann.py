import numpy as np
import random
import matplotlib.pyplot as plt
import scipy.stats
import sys

class Network(object):

	########################################################################
	# Initialise class variables
	# eg [2,5,4] would be a 3 layer network with 2 input noted, 5 hidden nodes and 4 output
	# weights are initialised with mean 0 variance 1
	########################################################################
	
	def __init__(self, sizes):

		np.random.seed(1)    #only used when repeatability is needed
		self.num_layers = len(sizes)
		self.sizes = sizes
		
		self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
		self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]
	
	
	########################################################################
	# Runs a set of inputs through the entire network, with current set of weights
	# a is np.array of training inputs of dimensions [ni, 1], eg [[0],[1],[0]]
	########################################################################
	
	def run(self, a):
		for b, w in zip(self.biases, self.weights):
			a = sigmoid(np.dot(w, a)+b)
		return a

		
	########################################################################
	# Implementation of back-propogation by gradient descent
	#-----------------------------------------------------------------------
	# http://neuralnetworksanddeeplearning.com/chap1.html
	# http://neuralnetworksanddeeplearning.com/chap2.html
	# https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/
	# 	a is np.array of training inputs of dimensions [ni, n_train]
	# 	y is np.array of training outputs of dimensions [no, n_train]
	# 	eta is the learning rate
	# 	err is the threshold error (break if below)
	# 	epochs is the max number of epochs to run
	########################################################################
	
	def train(self, inputs, y, eta, err, epochs):

		cost = []
		for epoch in range(epochs):
	
			z = []
			a = [inputs]
			dC_by_db = [np.zeros(b.shape) for b in self.biases]
			dC_by_dw = [np.zeros(w.shape) for w in self.weights]
				
			# forward propogation
			for b, w in zip(self.biases, self.weights):
				z.append(np.dot(w, a[-1]) + b)   # z = weighted layer inputs (ie pre-activation)
				a.append(sigmoid(z[-1]))		 # a = activated layer outputs
			
			C = np.array(0.5*np.square(a[-1] - y)).sum()   # C = total Cost
			if (C < err):
				break
			
			# backward propogation
			d = (a[-1] - y)	* sigmoid_prime(z[-1])   # d = dC/da = dC/dz*dz/da
			dC_by_dw = np.dot(d, a[-2].transpose())  # da/dw = a[-2] so dC/dw = d*a[-2]
			self.weights[-1] -= eta * dC_by_dw
			
			for layer in range(self.num_layers-3,-1,-1):
				d = np.dot(self.weights[layer+1].transpose(), d) * sigmoid_prime(z[layer])
				dC_by_dw = np.dot(d, a[layer].transpose())	
				self.weights[layer] += -eta * dC_by_dw
				cost.append(C)
		
		if (epoch == epochs-1):
			print('network did not converge\n', self.weights)
		else:
			print('network converged in ', epoch, ' epochs\nfinal weights:\n', self.weights)
		plt.plot(range(0,len(cost),1), cost)
		plt.show()
	
	
#####################################################
# Normalisation functions
# normalise set to have mean = 0, covariance = 1
# ref: http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf
#####################################################
 
def covariant(self, arr):
	# calculate covariance of array wrt itself
	return sum([x**2 for x in arr])/(arr.length+1)
   
def mean(self, arr):
	# calculate mean of array
	return sum([x for x in arr])/(arr.length+1)
   
def normalise(self, arr, factors):
	# apply normalisation factors to given array
	return (arr - factors.shift) * factors.scale
 
def denormalise(self, arr, factors):
	# de-apply normalisation factors to a given array
	return arr/factors.scale + factors.shift
   
def factors(self, arr):
	# calculate shift and scale factors for provided array
	arr_mean = self.mean(arr)
	return {
		"scale": (1/self.covariant(arr - arr_mean))**0.5,
		"shift": arr_mean
	}


#	def norms(self, inputs, outputs):
#    ninputs = []    
#    self.factors = []
#    for i in range(len(inputs)):
#        factors.append(self.factors(inputs))
#        ninputs.append(normalise(inputs, factors[-1]))
#    return ninputs  
 

   
    
#####################################################
# activation functions
# ref: http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf
# ref: https://towardsdatascience.com/activation-functions-neural-networks-1cbd9f8d91d6
#####################################################

def sigmoid(z):
    return 1/(1+np.exp(-z))
   
def sigmoid_prime(z):
    return sigmoid(z)*(1-sigmoid(z))
	
def hyperbolic_tangent(self, z):
    return 1.7159 * np.tanh(2/3 * z)
   
def hyperbolic_tangent_prime(self, z):
    return xx
   
def relu(self, z):
    return 0 if (z < 0) else z
   
def relu_prime(self, z):
    return 0 if (z < 0) else 1
   
def softplus(self, z):
    return np.ln(1+np.exp(z))
   
def softplus_prime(self, z):
    return 1/(1+np.exp(-z))
   
    
#####################################################
# test functions for plotting stuff
#####################################################
 
def plot_activations(self, act_fun, act_fun_prime):
   
    np = 50
    xmin = -10
    xmax = 10 
    x = [x*float(xmax-xmin)/float(np-1) for x in range(np)]   
 
    plt.plot(x, act_fun(x))
    plt.plot(x, act_fun_prime(x) )
    plt.show()
 
def plot_random(self):
    # plot histogram of randomly generated numbers, to test
    # weight generator
    mean = 0
    sd   = 0.5
    bins = 50
    np   = 1000000
   
    x = scipy.stats.norm(mean, sd)
    y = x.rvs(np)
   
    print("mean = ", np.mean(y))
    print("sd   = ", np.std(y) )
   
    plt.hist(y, bins)
    plt.show()
