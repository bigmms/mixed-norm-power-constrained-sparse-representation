import os
import cv2
import utils as PCSR
import numpy as np
import scipy.io as sio

rootdir = "./TestingImgs/"
extension = ".jpg"
IMAGE_SIZE = 256
PATH_SIZE = 16
# learns a dictionary with 100 elements
param = { 'K' : 256,
          'lambda1' : 0.5, 'numThreads' : 8, 'batchsize' : 256,
          'iter' : 10}
L = 10
eps = 1.0
mu = 1.
theta = 1.
kappa = 1.
beta = 10.
eta = 0.33
zeta = 1.
gamma = 2.2

ITER_TIMES = 3
Dict = sio.loadmat('./Data/Dict.mat')
Dict = Dict['Dict']

if __name__ == '__main__':
    for parent, dirnames, filenames in os.walk(rootdir):
        for filename in filenames:
            f,ext = os.path.splitext(rootdir+filename) # Split filename and type
            if ext == extension:
                im = cv2.imread(rootdir+filename, 3)
                im_ycbcr = cv2.cvtColor(im, cv2.COLOR_BGR2YCR_CB)
                Y = im_ycbcr[:, :, 0]
                U = Y
                M = Y

                for iter in range(ITER_TIMES):
                    print("process image... %s at iter %sth" % (rootdir + filename, str(iter+1)))
                    DA = PCSR.getAlpha_mpg(Y, Dict)
                    W = PCSR.getW(M, theta, mu)
                    U = PCSR.getU(mu, kappa, M, W)
                    M = PCSR.getM(beta, eta, zeta, kappa, gamma, Y, DA, U)
                im_ycbcr[:, :, 0] = M
                cv2.imwrite("./Results/" + filename[:-4] + '_PCSR.jpg', np.uint8(cv2.cvtColor(im_ycbcr, cv2.COLOR_YCR_CB2BGR)))
