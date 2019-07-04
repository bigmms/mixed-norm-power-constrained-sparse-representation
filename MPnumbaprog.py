from __future__ import division
import numpy as np
from numbapro import cuda
import numbapro.cudalib.cublas as cublas
from numba import *
import math

@cuda.jit('void(f4[:,:],i8[:,:],i8)')
def removeWinners(curCoef,winners,jj):
    i = cuda.grid(1)
    for k in xrange(jj-1):
        curCoef[i,winners[k,i]] = 0.

@cuda.jit('void(f4[:,:],f4[:,:],f4[:,:],i8[:,:],i8)')
def maxCoefsABS(curCoefs,coefs,coefsd,winners,k):
    i = cuda.grid(1)
    #This is not a great idea. Does cuda do inf? What is largest negative number?
    maxVal = -1.
    maxLoc = -i
    length = curCoefs.shape[1]
    for jj in xrange(length):
        if math.fabs(curCoefs[i,jj]) > maxVal:
            maxVal = math.fabs(curCoefs[i,jj])
            maxLoc = jj
    winners[k,i] = maxLoc
    coefs[i,maxLoc] = curCoefs[i,maxLoc]
    coefsd[i,maxLoc] = curCoefs[i,maxLoc]

def mp(dictionary,stimuli,k=None,minabs=None):
    """
    Does matching pursuit on a batch of stimuli.
    Args:
        dictionary: Dictionary for matching pursuit. First axis should be dictionary element number.
        stimuli: Stimulus batch for matching pursuit. First axis should be stimulus number.
        k: Sparseness constraint. k dictionary elements will be used to represent stimuli.
        minabs: Minimum absolute value of the remaining signal to continue projection. If nothing is given, minabs is set to zero and k basis elements will be used.
    Returns:
        coeffs: List of dictionary element coefficients to be used for each stimulus.
    """
    if k is None:
        k = dictionary.shape[0]
    if minabs is None:
        minabs = 0.

    bs = cublas.Blas()

    numDict = dictionary.shape[0]
    numStim = stimuli.shape[0]
    dataLength = stimuli.shape[1]
    assert k <= numDict
    #Setup variables on GPU
    d_coefs = cuda.to_device(np.zeros(shape=(numStim,numDict),dtype=np.float32,order='F'))
    d_curCoef = cuda.to_device(np.zeros(shape=(numStim,numDict),dtype=np.float32,order='F'))
    d_coefsd = cuda.to_device(np.zeros(shape=(numStim,numDict),dtype=np.float32,order='F'))
    d_winners = cuda.to_device(np.zeros(shape=(k,numStim),dtype=np.int64,order='F'))
    d_delta = cuda.to_device(np.zeros_like(stimuli,dtype=np.float32,order='F'))
    d_coefsd = cuda.to_device(np.zeros(shape=(numStim,numDict),dtype=np.float32,order='F'))
    #Move args to GPU
    d_stim = cuda.to_device(np.array(stimuli,dtype=np.float32,order='F'))
    d_stimt = cuda.to_device(np.zeros_like(stimuli,dtype=np.float32,order='F'))
    d_dict = cuda.to_device(np.array(dictionary,dtype=np.float32,order='F'))

    griddim1 = 32
    griddim2 = (32,32)
    assert numStim % 32 ==0 and dataLength % 32 == 0 and numDict % 32 == 0
    blockdimstim = int(numStim/griddim1)
    blockdim2 = (int(numStim/griddim2[0]),int(dataLength/griddim2[1]))
    blockdimcoef = (int(numStim/griddim2[0]),int(numDict/griddim2[1]))

    for ii in xrange(k):
        if minabs >= np.mean(np.absolute(d_stim.copy_to_host())):
            break
        bs.gemm('N','T',numStim,numDict,dataLength,1.,d_stim,d_dict,0.,d_curCoef)
        if ii > 0:
            removeWinners[griddim1,blockdimstim](d_curCoef,d_winners,ii)
        maxCoefsABS[griddim1,blockdimstim](d_curCoef,d_coefs,d_coefsd,d_winners,ii,0)
        #print d_winners.copy_to_host()
        bs.gemm('N','N',numStim,dataLength,numDict,1.,d_coefsd,d_dict,0.,d_delta)
        #print 'delta'
        #print d_delta.copy_to_host()
        #d_coefsd = cuda.to_device(np.zeros(shape=(numStim,numDict),dtype=np.float32,order='F'))
        bs.geam('N','N',numStim,numDict,0.,d_coefsd,0.,d_coefsd,d_coefsd)
        bs.geam('N','N',numStim,dataLength,1.,d_stim,-1.,d_delta,d_stim)
        #bs.geam('N','N',numStim,dataLength,1.,d_stimt,0.,d_delta,d_stim)
        #print 'stim'
        #print d_stim.copy_to_host()
    return d_coefs.copy_to_host()