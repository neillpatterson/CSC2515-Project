

import numpy as np
from numpy.random import multivariate_normal
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')
import seaborn as sns



mu1 = np.array([0, 1])
mu2 = np.array([-0.5, -1.0])
mu3 = np.array([3.2, .6])
mu4 = np.array([3, -1])

s1 = np.matrix([[1, 0.1], [0.1, 0.1]])
s2 = np.matrix([[0.3, 0.2], [0.2, 0.4]])
s3 = np.matrix([[0.5, 0.01], [0.01, 0.1]])
s4 = np.matrix([[0.5, -0.2], [-0.2, 0.2]])

d1 = multivariate_normal(mu1, s1, 500)
d2 = multivariate_normal(mu2, s2, 500)
d3 = multivariate_normal(mu3, s3, 500)
d4 = multivariate_normal(mu4, s4, 500)



fig, (ax1) = plt.subplots(1,1)

ax1.scatter(d1[:,0], d1[:,1], color='red')
ax1.scatter(d2[:,0], d2[:,1], color='blue')
ax1.scatter(d3[:,0], d3[:,1], color='green')
ax1.scatter(d4[:,0], d4[:,1], color='black')

plt.show()





sns.kdeplot(d1[:,0], d1[:,1], cmap='Blues', shade=True, shade_lowest=False)
sns.kdeplot(d2[:,0], d2[:,1], cmap='Greens', shade=True, shade_lowest=False)
sns.kdeplot(d3[:,0], d3[:,1], cmap='Reds', shade=True, shade_lowest=False)
sns.kdeplot(d4[:,0], d4[:,1], cmap='Purples', shade=True, shade_lowest=False)
#plt.show()



