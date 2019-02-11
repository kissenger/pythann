import ann
import numpy as np
import ast

# ===============================================================
# read input data file
# ===============================================================

LEARNING_RATE        = 1
EPOCH_LIMIT          = 100000
MAX_ERROR            = 0.0001

path = 'C:\\__FILES\\Gordon\\Neural Network\\'
f = open(path + "training_data.txt", 'r')

# get network dimensions
sizes=[int(x) for x in f.readline().split(',')]
print('network definition: ' + str(sizes))


# get training data
# train = aray containing all input data from file, except first line]
# ni = number of node in the input layer
# tin = [ni x n training data] array containing training inputs
# tout = [no x n training data] array containing training outputs
train=np.genfromtxt(f.readlines(), dtype=float, delimiter=',')
ni = sizes[0]
tin=np.array(train[0:,0:ni]).transpose();
tout=np.array(train[0:,ni:]).transpose();
print('training set:')
print(tin)
print(tout)


# ===============================================================
# train the network
# ===============================================================
  
network = ann.Network(sizes)
print('weights:')
print(network.weights)
network.train(tin, tout, LEARNING_RATE, MAX_ERROR, EPOCH_LIMIT)

# ===============================================================
# run the network on finalised weights
# ===============================================================
while 1: 
	inp = input('Enter inputs (or any char to quit): ')
	try:
		inp = [[float(x)] for x in inp.split(',')]
	except:
		break
	print(network.output(inp))
	