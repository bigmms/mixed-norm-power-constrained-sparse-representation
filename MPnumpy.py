from __future__ import division
import numpy as np

def mp(dictionary,stimuli,k=None,minabs=None):
    """
    Does matching pursuit on a batch of stimuli.

    Args:
        dictionary: Dictionary for matching pursuit. First axis should be dictionary element number.
        stimuli: Stimulus batch for matching pursuit. First axis should be stimulus number.
        k: Sparseness constraint. k dictionary elements will be used to represent stimuli.
        minabs: Minimum absolute value of the remaining signal to continue projection. If nothing is given, minabs is set to zero and k basis elements will be used.

    Returns
        coeffs: List of dictionary element coefficients to be used for each stimulus.
    """
    if k is None:
        k = dictionary.shape[0]
    if minabs is None:
        minabs = 0.
        
    numDict = dictionary.shape[0]
    numStim = stimuli.shape[0]
    dataLength = stimuli.shape[1]
    # dataLength = 1;
    coefs = np.zeros(shape=(numStim,numDict))
    stim = np.copy(stimuli)

    assert k <= numDict
    winners = np.zeros(shape=(k,numStim))

    stimn = np.arange(numStim)
    for ii in xrange(k):
        if minabs >= np.mean(np.absolute(stim)):
            break
        curCoef = np.dot(stim,dictionary.T)
        if ii != 0:
            for jj in xrange(ii-1):
                curCoef[stimn,winners[jj].astype(np.int)] = 0.
        dictn = np.argmax(np.absolute(curCoef),axis=1)
        winners[ii] = dictn
        coefsd = np.zeros_like(coefs)
        coefsd[stimn,dictn] = curCoef[stimn,dictn]
        coefs[stimn,dictn] = curCoef[stimn,dictn]
        delta = np.dot(coefsd,dictionary)
        stim = stim-delta

    return coefs
