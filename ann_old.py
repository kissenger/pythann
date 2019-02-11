import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
import sys

# ===============================================================
# References
# ===============================================================
# https://www.python-course.eu/neural_networks.php
# https://www.python-course.eu/neural_networks_with_python_numpy.php
# https://enlight.nyc/projects/neural-network/
# ===============================================================
# globals
# ===============================================================
#N_HIDDEN_NODES       = 4
LEARNING_RATE        = 0.1
EPOCH_LIMIT          = 500
MAX_ERROR            = 0.001
SIGMOID_MULTIPLIER   = 0.01
 
# weight setting
NORM_MEAN = 0.5
NORM_SD   = 0.5
 
# variables
error_array = []
 
# ===============================================================
# functions
# ===============================================================
def sigmoid(z):
    return 1/(1+np.exp(-SIGMOID_MULTIPLIER * z))
   
def sigmoid_drv(z):
    return np.multiply(z, (1 - np.matrix(SIGMOID_MULTIPLIER * z)))

    
# ===============================================================
# read input data file
# ===============================================================
path="C:\\Users\\gtaylor3\\Documents\\__WORK\\python\\"
f = open(path + "training_data.txt", 'r')
 
# read two lines, discard the first as a comment and capture the second
 
#print(f.readline().split(':')[1])
#data = f.readline()
#print(float(f.readline().split(':')[1]))
ni = float(f.readline().split(':')[1])
nh = float(f.readline().split(':')[1])
no = float(f.readline().split(':')[1])
 
# get bulk training data
data = f.readlines()
train = np.genfromtxt(data, delimiter=",")
 
# parse into variables
nt = len(train)                            # number of training points
tin = np.c_[train[:,0:ni], np.ones(nt)]           # inputs including bias
tout = train[:,ni:]                               # outputs
 
if len(train[0]) != ni + no:
    sys.exit("data array not consistent with inputs ni and no")
 
print(tout)
# ===============================================================
# run network
# ===============================================================
 
ni = ni + 1
global_error = 0
 
# Initalise layer weights
x=scipy.stats.norm(NORM_MEAN, NORM_SD)
#wih = x.rvs((nh, ni))                    # weights for input  --> hidden layer
#who = x.rvs((no, nh))                    # weights for hidden --> output layer
wih = np.ones((nh,ni))
who = np.ones((no,nh))
 
#================================================================
# input layer            hidden layer               output layer
#              wih                         who
#     tin     ----->        yh          ------>        yo
#================================================================
 
for epoch in range(EPOCH_LIMIT):
 
    # --- Forward propogation phase ---
    yh = sigmoid( np.dot(tin, np.matrix(wih).T ))          # yh is value of hidden layer
    yo = sigmoid( np.dot(yh, who.T) )                      # yo is value of output layer
   
    # --- Backward propogation phase ---
    cost = np.matrix(0.5*np.square(yo-tout)).sum()
    #yo_error = np.matrix(tout - yh).sum()
    yo_delta = np.dot(yo_error, sigmoid_drv(yo))   # --> array size [ o_nodes x training sets ]
    yh_error = np.dot(yo_delta, who)        # --> array size [ h_nodes x training sets ]
    yh_delta = np.multiply(yh_error, sigmoid_drv(yh))   # --> array size [ h_nodes x training sets ]
   
    wih += LEARNING_RATE *np.dot(np.matrix(yh_delta).T, tin)
    who += LEARNING_RATE *np.dot(np.matrix(yo_delta).T, yh)
    error_array.append(yo_error)

    print(yo_error)
    print(wih)

plt.plot(range(0,len(error_array),1), error_array)
plt.show()

# ===============================================================
# post-process
# ===============================================================