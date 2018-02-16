import torch
from torch.autograd import Variable
import numpy as np


def makeHawkesReconstructionDict(waveArrayTensor, parameterTensor=None):
    hawkesReconstructionDict, hawkesDict = {}, {}
    for event in waveArrayTensor.keys():
        if parameterTensor is None:
            parameter = Variable(torch.ones(1).double(), requires_grad=True)
        else:
            parameter = parameterTensor[event]
        if waveArrayTensor[event] is None or 'wavelet' not in waveArrayTensor[event]:
            print('makeHawkes...: ' + str(event) + ' not in wAT. Skipping')
            continue
        hawkesReconstructionDict[event] = torch.zeros(waveArrayTensor[event]['wavelet'].size())
        hawkesDict[event] = parameter
    resetHawkesTensors(hawkesReconstructionDict, hawkesDict, waveArrayTensor)
    return hawkesReconstructionDict, hawkesDict



def resetHawkesTensors(hawkesReconstructionDict, hawkesDict, waveArrayTensor):
    '''
    Uses a stepwise hazard-matching approximation to the exponential distribution given rate parameter in hawkesDict.
    Backward step takes care of updating the hawkesDict parameters.
    '''
    for event in hawkesReconstructionDict.keys():
        if event not in hawkesDict:
            print('Error (resetHawkesTensors): ' + str(event) + ' not found in hawkesDict. Skipping')
            continue
        if event not in waveArrayTensor or waveArrayTensor[event] is None or 'x' not in waveArrayTensor[event]:
            print('Error (resetHawkesTensors): ' + str(event) + ' not found in waveArrayTensor. Skipping')
            continue
        xs = waveArrayTensor[event]['x']
        if (xs > 0).data.sum() > 0:
            print('Error waveArrayTensor is non-causal at ' + str(event) + '. Skipping')
            continue
        xslb = -xs[1:]  # positive, closer to 0
        xsub = -xs[:-1]  # positive, further from 0
        parameter = hawkesDict[event]
        # CDF matching: value * (xsub - xslb) = CDF_ub - CDF_lb
        #               value = (CDF_ub - CDF_lb) / (xsub - xslb)
        #                     = ( (1-e^{-pu}) - (1-e^{-pl}) ) / ...
        #                     = e^{-pl} - e^{-pu} / ... 
        # Note: waveArrayTensor goes from most negative to least negative. Also, if the sign is wrong, relu will correct it.
        hawkesReconstructionDict[event] = ((-xslb*parameter).exp()-(-xsub*parameter).exp()) / (xsub-xslb)
        hawkesReconstructionDict[event][((xsub-xslb) == 0)] = 0
        # this overwriting mandates a forward pass before a backward
    return
