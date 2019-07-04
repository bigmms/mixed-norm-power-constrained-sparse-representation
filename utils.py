from __future__ import division
import spams
import numpy as np
import MPnumbaprog as mpg
import scipy.io as sio
import math


def getDictionary(Img, patch_size, **param):
    I = np.array(Img) / 255.
    A = np.asfortranarray(I)
    rgb = False
    X = spams.im2col_sliding(A, patch_size, patch_size, rgb)
    X = im2col(Img, (patch_size, patch_size))
    X = X - np.tile(np.mean(X, 0), (X.shape[0], 1))
    X = np.asfortranarray(X / np.tile(np.sqrt((X * X).sum(axis=0)), (X.shape[0], 1)), dtype=float)
    D = spams.trainDL(X, **param)
    return D


def getAlpha(Img, D, eps):
    PATCh_SIZE = 16
    X = im2col(Img, (PATCh_SIZE, PATCh_SIZE))
    X = np.asfortranarray(X)
    numThreads = -1
    alpha = spams.omp(X, D,  eps=eps, return_reg_path=False, numThreads=numThreads)
    return alpha


def im2col(mtx, block_size):
    mtx_shape = mtx.shape
    sx = mtx_shape[0] - block_size[0] + 1
    sy = mtx_shape[1] - block_size[1] + 1
    result = np.empty((block_size[0] * block_size[1], sx * sy))

    for i in range(sy):
        for j in range(sx):
            result[:, i * sx + j] = mtx[j:j + block_size[0], i:i + block_size[1]].ravel(order='F')
    return result


def col2im(mtx, image_size, block_size):
    p, q = block_size
    sx = image_size[0] - p + 1
    sy = image_size[1] - q + 1
    result = np.zeros(image_size)
    weight = np.zeros(image_size)
    col = 0
    for i in range(sy):
        for j in range(sx):
            result[j:j + p, i:i + q] += mtx[:, col].reshape(block_size, order='F')
            weight[j:j + p, i:i + q] += np.ones(block_size)
            col += 1
    return result / weight


def psnr(target, ref):
    target_data = np.array(target, dtype=np.float64)
    ref_data = np.array(ref, dtype=np.float64)
    diff = ref_data - target_data
    # print(diff.shape)
    diff = diff.flatten('C')
    rmse = math.sqrt(np.mean(diff ** 2.))
    return 20 * math.log10(255 / rmse)


def getAlpha_mpg(M, Dict): # GPU
    PATCh_SIZE = 16
    IMAGE_SIZE = 256
    kin = 32
    minabsin = 0

    X = im2col(M, (PATCh_SIZE, PATCh_SIZE))
    X = X.transpose()
    A = np.empty([58081, 256])

    A_gpu1 = mpg.mp(Dict.transpose(), X[0:32000, :],     k=kin, minabs=minabsin)
    A_gpu2 = mpg.mp(Dict.transpose(), X[32000:58080, :], k=kin, minabs=minabsin)
    A[0:32000, :] = A_gpu1
    A[32000:58080, :] = A_gpu2

    Alpha = np.dot(A, Dict.transpose())
    A[58080, :] = X[58080, :]
    result = col2im(Alpha.transpose(), [IMAGE_SIZE, IMAGE_SIZE], (PATCh_SIZE, PATCh_SIZE))
    return result


def grad(M):
    Mx = np.ones((256, 256), dtype=np.float64)
    My = np.ones((256, 256), dtype=np.float64)
    Mx[255, :] = M[0, :]
    Mx[0:254, :] = M[1:255, :]
    My[:, 255] = M[:, 0]
    My[:, 0:254] = M[:, 1:255]
    ux = Mx - M
    uy = My - M
    return ux, uy


def getW(M, theta, mu):
    ux, uy = grad(M)
    grad_u = np.sqrt(ux ** 2 + uy ** 2)  # || total variation || 2 ^ 2
    tmp = np.ones((256, 256)) * (-theta / mu)
    tmp2 = grad_u - tmp
    W = np.zeros((256, 256), dtype=np.float64)
    for i in range(256):
        for j in range(256):
            if tmp2[i, j] > 0:
                W[i, j] = tmp2[i, j]
            if grad_u[i, j] == 0:
                grad_u[i, j] = 1
    W = W / grad_u
    return W


def div(Px, Py):
    Px_tmp = np.ones((256, 256), dtype=np.float64)
    Py_tmp = np.ones((256, 256), dtype=np.float64)
    Px_tmp[0, :] = Px[255, :]
    Px_tmp[1:255, :] = Px[0:254, :]
    Py_tmp[:, 0] = Py[:, 255]
    Py_tmp[:, 1:255] = Py[:, 0:254]
    fx = Px - Px_tmp
    fy = Py - Py_tmp
    fd = fx + fy
    return fd


def getU(mu, kappa, M, W):
    I = np.eye(256, dtype=int)
    dg = sio.loadmat('./Data/dg.mat')
    dg = dg['dg']
    ux, uy = grad(M)
    q1 = np.multiply(ux, W)
    q2 = np.multiply(uy, W)
    div_w = div(q1, q2)
    A = np.multiply(mu, dg) + np.multiply(kappa, I)
    B = np.multiply(-mu, div_w) + np.multiply(kappa, M)
    AI = np.linalg.inv(A)
    U = np.linalg.lstsq(A, A)[0]
    return U


def Mfunc(beta, eta, zeta, kappa, gamma, Y, DA, U, m):
    return (beta/2) * ((Y - m)**2) + (eta/2) * (m**gamma) + (zeta/2)*(m - DA)**2 + (kappa/2)*(U - m)**2


def Min_indx_in_Mfunc(beta, eta, zeta, kappa, gamma, Y, DA, U):
    tmp_value = 99999999
    for i in range(0,255):
        current_value = Mfunc(beta, eta, zeta, kappa, gamma, Y, DA, U, i)
        if current_value > tmp_value:
            return i-1
        tmp_value = current_value
    return -1


def getM(beta, eta, zeta, kappa, gamma, Y, DA, U):
    M = np.zeros((256, 256), dtype=np.float64)
    for i in range(256):
        for j in range(256):
            M[i, j] = Min_indx_in_Mfunc(beta, eta, zeta, kappa, gamma, Y[i, j], DA[i, j], U[i, j])
    return M

