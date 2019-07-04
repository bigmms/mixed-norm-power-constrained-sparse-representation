import spams
import numpy as np
np.random.seed(0)
import time
print('test omp')
X = np.asfortranarray(np.random.normal(size=(64,100000)),dtype= float)
D = np.asfortranarray(np.random.normal(size=(64,200)))
D = np.asfortranarray(D / np.tile(np.sqrt((D*D).sum(axis=0)),(D.shape[0],1)),dtype= float)
L = 10
eps = 1.0
numThreads = -1

alpha = spams.omp(X, D, L=L, eps= eps,return_reg_path = False,numThreads = numThreads)

print 'D', np.shape(D)
print 'X', np.shape(X)
print 'Alpha', np.shape(alpha)
########################################
# Regularization path of a single signal
########################################
