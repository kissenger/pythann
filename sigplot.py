import numpy as np
import matplotlib.pyplot as plt


# ===============================================================
# Tests sigmoid and sigmoid derivative functions
# ===============================================================

SIGMOID_MULTIPLIER = 1
N_POINTS = 50
XMIN = -10
XMAX = 10

# ===============================================================
# Sigmoid functions to test
# ===============================================================

def sigmoid(z):
    return 1/(1+np.exp(-SIGMOID_MULTIPLIER * z))

def sigmoid_drv(z):
    s = 1/(1+np.exp(-SIGMOID_MULTIPLIER * z))
    print('================')
    print(s)
    return np.multiply(s, (1 - s))
     
# ===============================================================
# Create x array and call functions
# ===============================================================
     
xarr = np.array([])
for point in range(N_POINTS):
    xarr = np.append(xarr, float(XMAX - XMIN)/float(N_POINTS-1) * point + XMIN)
   
sig = sigmoid(xarr)
sigd = sigmoid_drv(xarr)
 
# ===============================================================
# Print and plot
# ===============================================================
  
print(xarr)
print(sigd)
print(sig)
plt.plot(xarr, sigd)
plt.plot(xarr, sig)
plt.show()

 