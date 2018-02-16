# Code to utilize the real-valued mark as feature
# Code to compute the mixture of functions defined on different intervals

import torch
from torch import nn
from torch.autograd import Variable
import numpy as np
import os
from numpy import genfromtxt
import pandas as pd
import re
import sys
from scipy import sparse
import pdb
import jcw_pywavelets
import pickle
import gzip

# seqs = np.tile([0,20,0.5],(N,1)) # replace with sample-specifics lbubs
# pyseqs = np.tile(np.linspace(5.5,12.2,40),(M,1))  # replace with sample-specific dwr spacings
# tts = np.zeros((N, Tmax))
# for i in range(N):
#     tts[i, 0:np.random.randint(Tmax)] = 1  # replace 1 with tts
# recons = np.ones((M, pyseqs.shape[1]))  # replace 1 with dwr;
# (after having used matrix multiply layer to get recons using DWR matrices and coefs.)

# print seqs.shape, pyseqs.shape # print tts # print recons.shape # print seqs

# maybe precalculate with vectors Pr contribution to time in seqs
# (based on target times and seqs) and store in (sparse) matrix?
# This can be computed without gradient updates:
# ct = getContribTensor(seq, tts, pyseq)  # Output (N, M, Tmax, Seqmax), require_grad=False
# overlay( recon.mm(ct), reducers )  # Output (N, M, Seqmax, Reducers)


# Load data from R
def loadPricklyData(path, dataShape):
    N, Tmax, E = dataShape
    
    eventdirs = os.listdir(path)
    timesValuesDict = {}
    for event in eventdirs:
        if os.path.isfile(path+event):
            continue
        ptsTimesValues = {}
        ptFiles = os.listdir(path+'/'+event)
        for ptFile in ptFiles:
            ptsTimesValues[ptFile] = genfromtxt(path+'/'+event+'/'+ptFile,
                                                delimiter=',', skip_header=1)
        timesValuesDict[event] = ptsTimesValues
        # pp.pprint(timesValuesDict.iterkeys().next())

    # Organize data into usable format
    waveArray = {}
    for event in timesValuesDict.keys():
        for filename in timesValuesDict[event].keys():
            if filename == 'WaveInfo.csv':
                waveArray[event] = timesValuesDict[event][filename]
                if np.any(waveArray[event][:,0] > 0):
                    # pdb.set_trace()
                    print('Warning, event ' + event + ' has non-causal spread')
                    print(waveArray[event][:,0])
                    waveArray[event][:,0] = np.linspace(waveArray[event][:,0].min(),0,waveArray[event].shape[0])
                    print('mapped to ')
                    print(waveArray[event][:,0])
    ptsTimesValues = {}
    for event in timesValuesDict.keys():
        ptsTimesValues[event] = []
        for filename in timesValuesDict[event].keys():
            if filename != 'WaveInfo.csv':
                ptsTimesValues[event].append([filename, timesValuesDict[event][filename]])
    # pp.pprint(ptsTimesValues['POC'][0])

    # Declare numpy variables, coercing overly large arrays by subsampling
    target = pd.read_csv(path+'/target.csv')
    target = target['target'][0]

    lbubMatrix = pd.read_csv(path+'/lbub.csv').as_matrix()
    lbub = {}
    for l in np.arange(lbubMatrix.shape[0]):
        lbub[l] = lbubMatrix[l, 1:3]
        events = waveArray.keys()
    # print events

    waveArrayTensor = {}
    for event in waveArray.keys():
        waveArrayTensor[event] = {'x': Variable(torch.from_numpy(np.array(waveArray[event])[:, 0]).double(),
                                                requires_grad=False),
                                  'wavelet': Variable(torch.from_numpy(np.array(waveArray[event][::-1, :])[1:, 2]).double(),
                                                      requires_grad=True)}
    # print waveArrayTensor['POC']
    ptsTimesValuesTensor = {}
    for event in ptsTimesValues.keys():
        tvArray = np.full((N, Tmax, E), np.nan)
        for ptFile, tv in ptsTimesValues[event]:
            index = int(re.match("pt([0-9]*)Times", ptFile).group(1))
            if index >= N:
                continue
            if len(tv.shape) == 1:
                tvArray[index-1, 0, :] = np.array(tv)
            else:
                if tv.shape[0] > Tmax:
                    temp = np.array(tv)
                    temp = temp[np.random.choice(temp.shape[0], size=Tmax, replace=False), :]
                    tvArray[index-1, :, :] = temp
                else:
                    temp = np.array(tv)
                    tvArray[index-1, 0:temp.shape[0], :] = temp
        ptsTimesValuesTensor[event] = Variable(torch.from_numpy(tvArray).type(torch.DoubleTensor), requires_grad=False)

    # print ptsTimesValuesTensor['HgbA1C']
    return target, events, lbub, ptsTimesValuesTensor, waveArrayTensor


def makeHistogramDecomposition(eventTimesValuesTensor, targetTimesTensor, bins=(32,128), causal=True, ranges=None, logWhenPossible=True, hd=None):
    '''
    Takes eTVT (N x Tmax x E) and tTT(N x Tmax x E) and returns the timeBased
    crossCorrelation 2d decomposition. Range setting overrides causal parameter.

    Note input: bins (scaleBins, timeBins)

    Note output: wdcp has scaleBins rows and timeBins columns
    Note fail output: wdcp is None and so are scaleBins and timeBins
    '''
    
    # Convert out of Variables
    eTVT = eventTimesValuesTensor.data.numpy()
    tTT = targetTimesTensor.data.numpy()
    N, Tmax, E = tTT.shape
    if tTT.shape[2] != 1:
        tTT = tTT[:,:,0]

    # Do computations. Can do sparsely in scipy if this becomes problematic
    dTV = eTVT[:,:,0][:,None,:] - tTT[:,:,None]  # diffs: N x Tmax x Tmax
    vTV = np.repeat(eTVT[:,:,1][:,None,:], Tmax, axis=1)  # values expanded: N x Tmax x Tmax
    dVTV = np.concatenate((dTV[:,:,:,None], vTV[:,:,:,None]), axis=3)  # N x Tmax x Tmax x 2
    valued = dVTV.transpose((3,1,0,2)).reshape(E, N*Tmax*Tmax)  # E x -1
    valued = valued[:,~np.isnan(valued[0,:])]
    valued = valued[:,~np.isnan(valued[1,:])]  # necessary?

    # Create 2d histogram
    if valued.shape[1] == 0:
        return None, None, None  # pass
    v0min, v0max, v1min, v1max = valued[0,:].min(), valued[0,:].max(), valued[1,:].min(), valued[1,:].max()
    if ranges is None:  # set ranges
        if causal:
            xmax = np.minimum(0.,v0max)
            if(v0min > xmax):
                return Variable(torch.zeros(1,1).double(), requires_grad=True),\
                    Variable(torch.ones(1).double(), requires_grad=False), \
                    Variable(torch.ones(1).double(), requires_grad=False)
        else:
            xmax = v0max
        if v1min == v1max:
            print('Warning, only one value detected in image construction (event value). Setting range at +/- 1 of value')
            v1min = v1min - 1
            v1max = v1max + 1
        if v0min == v0max or v0min == xmax:
            print('Warning, only one value detected in image construction (time). Setting range at +/- 1 of value')
            v0min = v0min - 1
            v0max = v0max + 1
            if causal:
                v0max = np.minimum(0, v0max)
                xmax = v0max
            if v0min >= v0max:
                v0min = v0max - 1
        ranges = [[v1min, v1max], [v0min, xmax]]
        # pdb.set_trace()
    else:
        if causal:
            ranges = np.array(ranges).copy()
            ranges[1,1] = np.minimum(0,ranges[1,1])  # apply causality
        
    if logWhenPossible and (valued.shape[1] > 0 and valued[1,:].min() > 0):
        bins = [np.power(2, np.linspace(np.log2(v1min),
                                        np.log2(v1max),bins[0]+1)),
                np.linspace(v0min,xmax,bins[1]+1)]
        # pdb.set_trace()
    if type(bins[1]) is np.ndarray:
        # print bins[1][1:]-bins[1][:-1]
        if np.any(bins[1][1:]-bins[1][:-1] < 0):
            pdb.set_trace()
        if len(bins[1]) == 1:
            print('Warning, time bins input was 0. returning image None')
            return None, None, None
    if type(bins[1]) is not np.ndarray:
        if bins[1] == 0:
            return None, None, None
    # print(bins, v0max, v0min, xmax)
    try:
        counts, vs, ts = np.histogram2d(valued[1,:],valued[0,:],bins=bins, range=ranges)  # TODO add weights
        counts = counts/len(tTT.flatten()[~np.isnan(tTT.flatten())])
    except ValueError:
        print('Warning: despite checks, attempted to make histogram')
        print(valued[1,:], valued[0,:])
        return None, None, None

    if hd is None:
        hd = jcw_pywavelets.create_haar_dictionary(np.maximum(np.log2(bins[0].shape[0],bins[1].shape[0])))
        hd = [Variable(torch.from_numpy(h),requires_grad=False) for h in hd]

    hdis = np.log2(counts.shape)
    wdcp = Variable(torch.from_numpy(counts).double(), requires_grad=False)
    wdcp = wdcp.t().matmul(hd[hdis[0]-1].t().cpu()).t().matmul(hd[hdis[1]-1].t().cpu())
    wdcp = Variable(wdcp.data, requires_grad=True)
    # decompose is over rows first, then columns -> reconstruct needs to be columns first, then rows.

    return wdcp, \
        Variable(torch.from_numpy(vs).double(), requires_grad=False), \
        Variable(torch.from_numpy(ts).double(), requires_grad=False)


# when you get a new event (e_i, t_i), you reconstruct the value space,
# element*wise with the bin the event falls into, then reconstruct the time space 
# and apply the proportional and the full effect back

# functions:
#   (1) data (time-space counts) -> 2d coefficients
#   (2) new time, value -> hazard function (proportional to density value)
#       this is just onehot * hazard image * (flipped) ?
#   existing mapping from function to hazard is already created
# variables:
#   a matrix per event: [coef, hd's, valuebin]--> image


def getWaveletSizes(events, waveArrayTensor):
    wsizes = []
    for event in events:
        if event not in waveArrayTensor or waveArrayTensor[event] is None:
            wsizes.append(0)
        else:
            wsizes.append(waveArrayTensor[event]['wavelet'].data.shape[0])
    return wsizes


def valueTensorToValueBinTensor(valueTensor, scaleBins):  # TODO allow scaleBins to be multi-D
    '''
    Get valueBins from values. Not for autograd.
    '''
    if scaleBins is None:
        return None
    
    npvt = valueTensor.data.numpy()
    npvt = npvt[~np.isnan(npvt)]  # only works if nans are at the top
    if scaleBins.data.numpy().shape[0] == 1:
        ndig = np.zeros(npvt.shape)
    else:
        ndig = np.digitize(npvt, scaleBins.data.numpy()[1:-1])  # new values that are extreme to go 0 and 2^n-1
    
    if ndig.shape[0] == 0:
        ndig = np.array(np.nan)
    return torch.from_numpy(ndig).long()


def getImageTensors(target, events, imageLengths, ptsTimesValuesTensor, scaleSteps):
    '''
    Generates images (w2ds), accessory information (bounds stepsValues and stepsTimes), and a
    mapping from pts to the image (valueBinIndices)
    '''
    dim2d = np.empty((len(events),2))
    for eventi, event in enumerate(events):
        dim2d[eventi,:] = \
            np.array([scaleSteps, imageLengths[eventi]])  # value steps, time steps
    w2ds = np.empty(len(events),np.ndarray)
    stepsValues = np.empty(len(events), np.ndarray)
    stepsTimes = np.empty(len(events), np.ndarray)
    for eventi, event in enumerate(events):
        w2ds[eventi] = Variable(torch.DoubleTensor(dim2d[eventi]), requires_grad=False)
        stepsValues[eventi] = Variable(torch.Tensor(dim2d[0]+1), requires_grad=False).double()
        stepsTimes[eventi] = Variable(torch.Tensor(dim2d[1]+1), requires_grad=False).double()
        # w2ds = Variable(torch.Tensor(len(events), dim2d[0], dim2d[1]), requires_grad=False).double()
    hd = jcw_pywavelets.create_haar_dictionary(10, Variable)
    for eventi, event in enumerate(events):
        if event in ptsTimesValuesTensor:
            # print(event)
            w2ds[eventi], stepsValues[eventi], stepsTimes[eventi] = \
                makeHistogramDecomposition(ptsTimesValuesTensor[event],
                                           ptsTimesValuesTensor[target],
                                           bins=dim2d[eventi],
                                           hd=hd)
        else:
            w2ds[eventi], stepsValues[eventi], stepsTimes[eventi] = None, None, None
    # N x event x Tmax
    N = ptsTimesValuesTensor[target].size()[0]
    valueBinIndices = np.empty(N, np.ndarray)
    for i in np.arange(N):
        valueBinIndices[i] = np.empty(len(events),np.ndarray)
        for eventi, event in enumerate(events):
            if event in ptsTimesValuesTensor:
                try:
                    valueBinIndices[i][eventi] = \
                        valueTensorToValueBinTensor(ptsTimesValuesTensor[event][i,:,1], stepsValues[eventi])
                except:
                    pdb.set_trace()
            else:
                valueBinIndices[i][eventi] = None
    # print w2ds
    # print w2ds[2].matmul(hd[np.log2(w2ds[2].data.numpy().shape)[1]-1]).\
    #     t().matmul(hd[np.log2(w2ds[2].data.numpy().shape)[0]-1]).t()  # this is the reconstruct instruction
    # plt.imshow(w2ds[2].matmul(hd[np.log2(w2ds[2].data.numpy().shape)[1]-1]).
    #            t().matmul(hd[np.log2(w2ds[2].data.numpy().shape)[0]-1]).t().data.numpy(),
    #            extent=(stepsTimes[2].data.numpy().min(),
    #                    stepsTimes[2].data.numpy().max(),
    #                    stepsValues[2].data.numpy().min(),
    #                    stepsValues[2].data.numpy().max()),origin='lower')
    # plt.show()

    return stepsValues, stepsTimes, w2ds, valueBinIndices, dim2d


def getOneHotValueBinTensor(valueBinTensor,bins):
    '''
    Takes LongTensor and returns a LongTensor onehot representation of it
    '''
    dims = len(valueBinTensor.shape)
    onehot = torch.zeros(tuple(np.append(valueBinTensor.numpy().shape,bins))).double()
    onehot.scatter_(dims,valueBinTensor.unsqueeze(dims),1)
    return onehot
    
    
# def valueBinVectorToHazards(onehotValueBinTensor, w2dEventTensor, hds, scalehd=None, timehd=None):
#     '''
#     Takes each value in onehotValueBinTensor and applies w2dEventTensor to get hazard function and
#     density estimate at value.
#     '''
#     if scalehd is None:
#         scalehd = hds[np.log2(w2dEventTensor.data.shape[0])-1]
#     if timehd is None:
#         timehd = hds[np.log2(w2dEventTensor.data.shape[1])-1]

#     # Take intermediate representation (
#     onehotValueBinTensor   # batch x Tmax x bins
#     return hazardTensor, deTensor  # (these are Variables)


def derivedTensors(lbub, target, ptsTimesValuesTensor, targetSteps, customWatx=None):
    # watx = waveArrayTensor[target]['x'].data.numpy()
    N = ptsTimesValuesTensor[target].size()[0]
    if customWatx is None:
        watx = np.empty(N, dtype=np.ndarray)
        for i in np.arange(N):
            watx[i] = np.linspace(lbub[i][0],lbub[i][1],num=targetSteps+1)
    else:
        watx = customWatx
    countsInWatxTensor = np.empty(N, dtype=Variable)
    distsInWatxTensor = np.empty(N, dtype=Variable)
    for i in np.arange(N):  # get counts in watx[i] window from ttimes (ptsTimesValuesTensor[target][i, :, 0])
        wonans = ptsTimesValuesTensor[target][i, :, 0].data.numpy()
        wonans = wonans[~np.isnan(wonans)]
        hist = np.histogram(wonans, bins=watx[i])
        countsInWatxTensor[i] = Variable(torch.from_numpy(hist[0]).double(), requires_grad=False)
        distsInWatxTensor[i] = Variable(torch.from_numpy(hist[1][1:]-hist[1][:-1]).double(),
                                        requires_grad=False)
    return watx, countsInWatxTensor, distsInWatxTensor


def shift(array, shift, fill_value=np.nan):
    result = np.roll(array, shift)
    if shift > 0:
        result[:shift] = fill_value
    elif shift < 0:
        result[-shift:] = fill_value
    return result


def sequenceToSequenceInterpolationMap(input, output, style='stepwise', ioSorted=True, applyCausalZeroing=True):
    ''' Returns a torch sparse matrix that maps input to output. Useful for backpropagation
    Vars:
    input - a vector of times
    output - a vector of times

    applyCausalZeroing zeros out the first output unit where input is relevant. this is if you want to do censoring
    on the output side. (you can also just hadamard the output side but you have to find where. this finds where.)
    note: to maintain causal timing, you should always 0 out the first affected output.
    note: if you don't care about causal timing, set applyCausalZeroing to False

    you could also apply causal zeroing on the input (e.g. censor is 0.7, output is [0,1,2], input is 0.1 steps starting at 0.5
      then input censoring will put nonzero values at 1.2, whereas output censoring will put nonzero values at 2.
      but this is problematic because 1.2 maps onto [1,2], so it is effectively a 0.5 censor not a 0.7 censor)

    Returns:
    a sparse matrix i x o such that matrix.t().mm(valuesAtInputs.t()) = valuesAtOutputs: a linear or stepwise interpolation
    (or valueAtInputs.mm(matrix) = valuesAtOutputs)
    '''

    # for each output two closest input indices
    if not ioSorted:
        print('not isSorted is not implemented')
        return

    # pdb.set_trace()
    if style == 'linear':  # linear interpolation of a linear interpolation # not a very good approximation
        diffm = np.subtract.outer(output, input)
        diffp = diffm.copy()
        diffp[diffp <= 0] = sys.float_info.max
        pis = np.argmin(diffp, axis=1)  # indices of minimally positive differences
        diffn = diffm.copy()
        diffn[diffm >= 0] = -sys.float_info.max
        nis = np.argmax(diffn, axis=1)
        
        pis[diffp[np.arange(len(output)), pis] == sys.float_info.max] = -1  # check for Infs
        nis[diffn[np.arange(len(output)), nis] == -sys.float_info.max] = -1  # check for Infs
        
        notm1 = ~((pis == -1) | (nis == -1))

        activeis = np.where(notm1)[0]

        # print activeis
        pcts = 1 - diffm[activeis, pis[notm1]] / (diffm[activeis, pis[notm1]] - diffm[activeis, nis[notm1]])
        return sparse.coo_matrix((np.concatenate([pcts, 1 - pcts]),
                                  (np.concatenate([pis[notm1], nis[notm1]]),
                                   np.concatenate([activeis, activeis]))),
                                 shape=(len(input), len(output)),
                                 dtype=np.float_)
    elif style == 'stepwise':
        # gives an i-1 x o-1 matrix whereas linear gives a i x o matrix for throughput into sparseSlicer
        # note sparseSlicer resolves this, but not smartly
        # stepwise maps stepwise hazards (preferred)
        # note: causality is unaccounted for here. causal censor can occur after this reconstruction by e.g. hadamard.
        # or more complicatedly, causal censor can occur before, but with 0ing of first nonzero steps before next output boundary
        # pdb.set_trace()
        reversed = False
        if input[0] > input[-1]:
            reversed = True
            input = input[::-1]
        outputDiff = output - shift(output,1,np.nan)  # first value is nan
        ubOindex = np.digitize(input, output)
        # lbOindex = shift(ubOindex, 1, 0)
        lbOindex = ubOindex - 1
        ubOindexLowerIgnores = (ubOindex - 1) < 0
        ubOindexCapped = ubOindex.copy()
        ubOindexCappedIgnores = ubOindexCapped >= len(output)
        ubOindexCapped[ubOindexCappedIgnores] = 0
        
        prevOutput = output[ubOindex-1]  # possibly getting corrupted 0 -> -1 values
        prevOutput[ubOindexLowerIgnores] = np.nan
        nextOutput = output[ubOindexCapped]  # possibly getting corrupted 0s
        nextOutput[ubOindexCappedIgnores] = np.nan

        sparseValues = []
        # pdb.set_trace()
        for i, bi in enumerate(lbOindex):  # jumping across columns
            # if bi == 0:
            #     continue
            if i > 0 and bi > ubOindex[i-1]:
                for k in np.arange(ubOindex[i-1], ubOindex[i]-1):
                    if k > 0:
                        sparseValues.append([i-1, k, 1])
            if ubOindex[i] < output.shape[0]:
                if i > 0 and i < output.shape[0]:
                    if len(sparseValues) > 0 and sparseValues[-1][:2] == [i-1, ubOindex[i]-1]:
                        pass  # already worked this interval
                    else:
                        sparseValues.append(
                            [i-1, ubOindex[i]-1, (input[i]-np.maximum(prevOutput[i],input[i-1]))/outputDiff[bi+1]])
                if i+1 < input.shape[0]:
                    sparseValues.append(
                        [i, ubOindex[i]-1, (np.minimum(nextOutput[i],input[i+1])-input[i])/outputDiff[bi+1]])
            # print(sparseValues)
        sv = np.array(sparseValues)
        if applyCausalZeroing and sv.shape[0] > 0:  # find min col remove all such elements (i.e. they will zeros)
            # print(sv)
            sv = sv[sv[:,1] != sv[:,1].min(),:]
            # print('becomes')
            # print(sv)
        if sv.shape[0] == 0:
            return sparse.coo_matrix((input.shape[0]-1,output.shape[0]-1), dtype=np.float_)
        svcoo = sparse.coo_matrix((sv[:,2], (sv[:,0],sv[:,1])), shape=(input.shape[0]-1,output.shape[0]-1))
        if not reversed:
            return svcoo
        else:
            return sparse.coo_matrix(np.flipud(svcoo.todense()))
    else:
        print('Error: not an accepted style. Use e.g. \'stepwise\' or \'linear\'')
        return None


def sparseSlicer(M):
    return sparse.coo_matrix(M.toarray()[:-1,:-1])  # chop off bottom and right.; while not fantastic, does protect causality in 'linear'
    # return sparse.coo_matrix(M.toarray()[1:,1:])  # chop off top and left


# test = sequenceToSequenceInterpolationMap(input=np.array([0, 0.5, 1., 1.4, 2., 4]),
#                                           output=np.array([0, 0.5, 1.5, 2.4, 3, 4.5]),
#                                           style='stepwise')
# # print(np.subtract.outer(np.array([0., 0.8, 1.6, 2.4, 3.]), np.array([1., 1.5, 2., 4.])))
# print(test.todense())
# print(sparseSlicer(sequenceToSequenceInterpolationMap(input=np.array([1., 1.5, 2., 4.]),
#                                           output=np.array([0, 0.8, 1.6, 2.4, 3]),
#                                           style='linear')).todense())
# print(test.todense())
# print(test.todense().dot(np.array([[1., 10., 100.]]).transpose()))

# # reversed, as we need to negate the cross-correlation
# test2 = sequenceToSequenceInterpolationMap(input=np.array([4., 2., 1.5, 1.]),
#                                            output=np.array([0., 0.8, 1.6, 2.4, 3]),
#                                            style='stepwise')
# # print(np.subtract.outer(np.array([0., 0.8, 1.6, 2.4, 3.]), np.array([1., 1.5, 2., 4.])))
# print(test2)
# print(test2.todense())
# print(test2.todense().dot(np.array([[1., 10., 100.]]).transpose()))


def getSSMap(i, j, ssDict, Tmax):
    ''' Returns selected sparse matrix in ssDict '''
    key = i*Tmax+j
    if key in ssDict:
        return ssDict[i*Tmax+j]
    else:
        return None


def scipyCooToTorchCoo(M):
    return torch.sparse.DoubleTensor(torch.LongTensor([M.row.tolist(),
                                                       M.col.tolist()]),torch.DoubleTensor(M.data))


def scipyToTorchDense(M):
    ''' Pytorch hasn't figured out sparse tensors yet, e.g. torch.mm(sparse V1,V2) throws an error.
    So in the meantime, let's just use memory dense mapping matrices and limit our problem size
    '''
    return Variable(torch.from_numpy(M.todense()).type(torch.DoubleTensor), requires_grad=False)


# Need ssmap of dimension: event x pt x Tmax ( x input x output )_sparse, i.e. shifted by eventtimes
# Since they're different sizes, maybe just make a dict of {event, Variable}
def getSSMaps(ptsTimesValuesTensor, waveArrayTensor, watx):
    ssmaps = {}
    events = waveArrayTensor.keys()
    for event in events:

        if event not in ptsTimesValuesTensor or ptsTimesValuesTensor[event] is None or waveArrayTensor[event] is None:
            continue
        # outer add pt times to waveArray sequence #pTX: pt x Tmax x wseq
        pTX = np.add.outer(ptsTimesValuesTensor[event].data.cpu().numpy()[:, :, 0],
                           -waveArrayTensor[event]['x'].data.cpu().numpy())  # flip wAT_i to positives and add to pTVT_i, then create ss mapping

        # pdb.set_trace()
        # print pTX
        # compute each sparse matrix
        ssmapDict = {}
        for i in range(pTX.shape[0]):  # N
            for j in range(pTX.shape[1]):
                if np.any(np.isnan(pTX[i, j, :])):
                    continue
                ssmapDict[i*pTX.shape[1]+j] = scipyToTorchDense(
                    sequenceToSequenceInterpolationMap(input=pTX[i, j, :],
                                                       output=watx[i],
                                                       style='stepwise',
                                                       applyCausalZeroing=True))
                # ssmapDict[i*pTX.shape[1]+j] = scipyToTorchDense(sparseSlicer(
                #     sequenceToSequenceInterpolationMap(input=pTX[i, j, :],
                #                                        output=watx[i],
                #                                        style='linear')))
        # to use stepwise, simply remove the sparseSlicer  # TODO check on this. looks like an index shift too
                
        # pytorch only does 2d sparseTensors effectively. Fortunately this is what I need.
        # Let's store them in ssmapDict and pull them out as necessary
        # Turns out they don't do these effectively either. Using dense for now
        
        # # make dense while pytorch figures out how to do sparseTensors
        # denseMap = np.full((pTX.shape[0], pTX.shape[1], pTX.shape[2], watx.shape[0]), np.nan)  # BIG!
        # for i in range(pTX.shape[0]):
        #     for j in range(pTX.shape[1]):
        #         if np.any(np.isnan(pTX[i, j, :])):
        #             continue
        #         denseMap[i, j, :, :] = ssmapDict[i*pTX.shape[1]+j].todense()
        
        # # convert dense tensor into a sparseTensor
        # denseMap[np.isnan(denseMap)] = 0
        # tDenseMap = torch.from_numpy(denseMap)
        # tIndices = torch.nonzero(tDenseMap)
        # print tDenseMap.size()
        # print tIndices.size()
        # if len(tIndices.size()) == 0:
        #     ssmap = torch.sparse.DoubleTensor(tDenseMap.size())
        # else:
        #     tIndices = tIndices.t()
        #     tValues = tDenseMap[tIndices[0], tIndices[1], tIndices[2], tIndices[3]]
        #     ssmap = torch.sparse.DoubleTensor(tIndices, tValues, tDenseMap.size())
        
        # pp.pprint(tIndices)
        # print denseMap.shape, tDenseMap.shape
        
        # create ssmaps[event] = Variable
        # print tValues.dtype
        # requires_grad=False
        # ssmaps[event] = ssmap
        ssmaps[event] = ssmapDict

    # DOTO bug in ssmaps: reproduced by:
    # torch.cat([ssmaps['Ketoacidosis'][k].sum(0) for k in ssmaps['Ketoacidosis'].keys()],0).view(-1,200).sum(0)
    # it's because waveArrayTensor is entirely negative!
    return ssmaps


def saveJcwModel(runStepParameters, path):
    with gzip.GzipFile(path+'.gz', 'w') as f:
        pickle.dump(len(runStepParameters),f)
        for r in runStepParameters:
            pickle.dump(r, f)
        print('Save complete: ' + path + '.gz')
        return f.close()


def loadJcwModel(path):
    # jcwList = pickle.load(os.open(path,'rb'))
    # os.close(path)
    with gzip.open(path, 'rb') as f:
        vals = []
        objs = pickle.load(f)
        for _ in range(objs):
            vals.append(pickle.load(f))
    return vals

    # reduce the mapped constructions

    # combine the events

    # transform as desired (identity for now)
    
    # transform as necessary to valid hazard: [1e-big, big)

    # compute loss
    # loss = (yv - wv.t().mm(xv)).pow(2).sum()
    # print(t, loss.data[0])

    # if t:
    #     wv.grad.data.zero_()
    
    # loss.backward()

    # wv.data += learningRate * wv.grad.data

    
def makeWvtCoefficientPenaltyVector(maxLength):
    coefs = np.empty(int(maxLength), np.float_)
    coefs[0] = np.power(2,-0.5)
    for i in np.arange(np.log2(maxLength)):
        coefs[int(np.power(2,i)):int(np.power(2,i+1))] = np.power(2, -i/2 - 1)
    coefs = 1/coefs  # inverted s.t. hadamard penalizes the highpasses
    coefs = Variable(torch.from_numpy(coefs).double(), requires_grad=False)
    return coefs


def attachPenalties(waveArrayTensor):
    ''' Overwrites the input by adding a coefficientPenalty to the wAT dictionary '''
    eventLengths = np.zeros(len(waveArrayTensor), dtype=int)
    for i, e in enumerate(waveArrayTensor.keys()):
        if waveArrayTensor[e] is not None:
            eventLengths[i] = len(waveArrayTensor[e]['wavelet'])
    maxLength = max(eventLengths)

    coefs = makeWvtCoefficientPenaltyVector(maxLength)
    for i, e in enumerate(waveArrayTensor.keys()):
        if waveArrayTensor[e] is not None:
            waveArrayTensor[e]['coefficientPenalty'] = coefs[:eventLengths[i]]
    return waveArrayTensor


def getPenalties2d(w2ds):
    ''' Return a list objects referencing penalty matrices for Hadamard '''
    p2ds = np.empty(len(w2ds),Variable)
    maxdim = 1
    for i, wi in enumerate(w2ds):
        if wi is not None and np.maximum(wi.size(),maxdim).max() > maxdim:
            maxdims = np.max(wi.size())
    coefs = makeWvtCoefficientPenaltyVector(maxdims)

    for i, wi in enumerate(w2ds):
        if wi is not None:
            dims = wi.size()
            p2ds[i] = torch.ger(coefs[:dims[0]],coefs[:dims[1]])
    return p2ds


def makeHadamardCensor(censorVector, waveArrayTensor):
    ''' Takes a vector of censor times and create hadamard matrices for \'wavelet\' in waveArrayTensor
    based on the \'x\' spacing. 
    '''
    if censorVector is None:
        return None
    if (censorVector > 0).data.sum() > 0:
        print('Warning: censorVector has values above zero, i.e. not causal. Unchecked')
    C = censorVector.size()[0]
    hadamardCensor = {'shape0': C}
    for event in waveArrayTensor.keys():
        if waveArrayTensor[event] is not None:
            wat = waveArrayTensor[event]['x'][1:]
            cmatrix = ((wat.unsqueeze(0).expand(C, wat.size()[0]) - censorVector.unsqueeze(1).expand(C, wat.size()[0])) < 0).double()
            if event in hadamardCensor:
                print('Error: ' + str(event) + ' already in hadamardCensor. Note \'shape0\' is protected. Skipping.')
            else:
                hadamardCensor[event] = cmatrix
    return hadamardCensor

    
class InterpolateLayer(nn.Module):
    def __init__(self, reconstruction, rxWxMap):
        ''' Maps reconstruction on wx:
        
        Parameters:
        reconstruction -- the wavelet reconstruction
        rxWxMap -- a map from the rx sequence to the wx sequence for linear interpolation: i.e. wx_i = k*rx_j + (1-k)*rx_k
        '''
        super(Linear, self).__init__()
        

        
    def forward(self,
                seq,
                pyseq,
                tt,
                recon):
        self.save_for_backward(seq, pyseq, tt, recon)
        
        
    
        return 0

    def backward(self, outTensor):
        seq, pyseq, tt, recon = self.saved_tensors 
        return (seq, pyseq, tt, recon)
    

class ReduceLayer(torch.autograd.Function):  # Surely this is already implemented
    def forward(self,
                seq,
                pyseq,
                tt,
                recon,
                reducers):
        self.save_for_backward(seq, pyseq, tt, recon, reducers)
        
        
        
        return 0

    def backward(self, outTensor):
        seq, pyseq, tt, recon = self.saved_tensors
        return (seq, pyseq, tt, recon, reducers)
