import scipy.io as sio
import numpy as np
from skimage import data, io
import cv2

M = sio.loadmat('M.mat')
M = M['M']
# cv2.imshow('M', M)
# cv2.waitKey(0)
#
# A = cv2.imread('kodim11.JPG', 3)
# print type(A)
# cv2.imshow('Aasd', A)
# cv2.waitKey(0)
Dict = sio.loadmat('Dict.mat')
Dict = Dict['Dict']

# normalized = np.zeros((256,256), dtype=float)
# min = np.minimum(Dict)
# max = np.maximum(Dict)
# for i in range(256):
#     for j in range(256):
#         normalized[i][j] = (Dict[i][j] - min) / (max - min)

cv2.imshow("Dict", Dict)
cv2.waitKey(0)
# def im2col(mtx, block_size):
#     mtx_shape = mtx.shape
#     print mtx_shape
#     sx = mtx_shape[0] - block_size[0] + 1
#     sy = mtx_shape[1] - block_size[1] + 1
#
#     result = np.empty((block_size[0] * block_size[1], sx * sy))
#
#     for i in range(sy):
#         for j in range(sx):
#             result[:, i * sx + j] = mtx[j:j + block_size[0], i:i + block_size[1]].ravel(order='F')
#     return result
#
# X = im2col(M, (16, 16))
# print type(X), np.shape(X)
