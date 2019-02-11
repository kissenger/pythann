import scipy.stats
import matplotlib.pyplot as plt
 
# ===============================================================
# Tests sigmoid and sigmoid derivative functions
# ===============================================================
 
NORM_MEAN = 0
NORM_SD   = 0.5
BINS = 50
 
# ===============================================================
# Create x array and call functions
# ===============================================================
  
x=scipy.stats.norm(NORM_MEAN, NORM_SD)
y = x.rvs(1000000)
 
plt.hist(y, bins=BINS)
plt.show()