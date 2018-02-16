
import pandas as pd
import numpy as np
import jcw_pywavelets
import torch
import pdb
from tqdm import tqdm
from torch.autograd import Variable


def makeCrossCorrelationTensor(eTVT, tTT, causal=True, bins=16, ranges=None, dim=1, hd=None):
    if tTT.shape[2] != 1:
        tTT = tTT[:,:,0]

    # Do computations. Can do sparsely in scipy if this becomes problematic
    N, Tmax, E = eTVT.shape
    if Tmax > 50:
        pdb.set_trace()
    dTV = eTVT[:,:,0][:,None,:] - tTT[:,:,None]  # diffs: N x Tmax x Tmax
    vTV = np.repeat(eTVT[:,:,1][:,None,:], Tmax, axis=1)  # values expanded: N x Tmax x Tmax
    dVTV = np.concatenate((dTV[:,:,:,None], vTV[:,:,:,None]), axis=3)  # N x Tmax x Tmax x 2
    valued = dVTV.transpose((3,1,0,2)).reshape(E, N*Tmax*Tmax)  # E x -1
    valued = valued[:,~np.isnan(valued[0,:])]
    # valued = valued[:,~np.isnan(valued[1,:])]  # this removes any non-numerics
    if valued.shape[1] == 0:
        return None, None, None
    if causal:
        xmax = np.minimum(0.,valued[0,:].max())
        if valued[0,:].min() > xmax:
            return None, None, None
    else:
        xmax = valued[0,:].max()

    if ranges is None:
        ranges = [valued[0,:].min(), xmax]
    if valued[0,:].min() >= xmax:
        ranges[0] = xmax-1  # causal applied to range
    if hd is None:
        if type(bins) == int:
            hdsize = bins
        elif type(bins) == np.array:
            hdsize = len(bins)
        else:
            print('Error in makeCrossCorrelationTensor bins parameter')
            return None, None, None
        hd = jcw_pywavelets.create_haar_dictionary(hdsize)
    # pdb.set_trace()
    if dim == 1:
        hist, histEdges = np.histogram(valued[0,:], bins=bins, range=ranges)
        hist = hist / np.count_nonzero(~np.isnan(valued[0,:]))
        hdis = np.log2(hist.shape)
        # htensor = Variable(torch.from_numpy(hist).double(), requires_grad=False)
        hwave = hist.dot(hd[hdis[0]-1].transpose())
        return hwave, histEdges, None
    return None, None, None


def timelineToWaveArrayTensors(target, ptsTimesValuesTensor, bins=16, ranges=None, causal=True):
    waveArrayTensor = {}
    # pdb.set_trace()
    tts = ptsTimesValuesTensor[target]
    for event in tqdm(ptsTimesValuesTensor.keys()):
        # if event == 'Person|GENDER_CONCEPT_ID:FEMALE|NA':
        #     pdb.set_trace()
        b = bins
        if bins is None:
            b = 16
        if type(bins) is dict:
            b = bins[event]
        r = ranges
        if type(ranges) is dict:
            r = ranges[event]
            
        cc, ccEdges, _ = makeCrossCorrelationTensor(ptsTimesValuesTensor[event], tts, causal=causal, bins=b, ranges=r,
                                                    hd=jcw_pywavelets.create_haar_dictionary(10))
        # print(cc, ccEdges)
        if cc is None or ccEdges is None:
            waveArrayTensor[event] = None
        else:
            waveArrayTensor[event] = {'x': Variable(torch.from_numpy(ccEdges), requires_grad=False),
                                      'wavelet': Variable(torch.from_numpy(cc), requires_grad=True)}
    return waveArrayTensor


def timelineToPricklyValues(dat4, target, events=None, lbubs=None, tvShape=None, eventBins=None, vectorType=Variable, ranges=None, useUnique=True, verbose=False):
    ''' Take 4-column pandas df (STUDY_ID, TIME, VALUE, EVENT) and create data for wavelet point process.
    eventBins can be None, a number, or a dict[event] = #bins, a power of 2
    '''

    # pdb.set_trace()
    if useUnique:
        dat4 = dat4.drop_duplicates()
    if (np.array([d in ['STUDY_ID','TIME','event','VALUE'] for d in dat4.columns])-1).sum() == 0:  # has correct columns
        dat4 = dat4[['STUDY_ID','TIME','event','VALUE']]
    else:
        print('Warning: unknown columns. Pulling first four columns and naming them STUDY_ID, TIME, event, and VALUE')
        dat4 = dat4.iloc[:,:4]
        dat4.columns = ['STUDY_ID','TIME','event','VALUE']
    if dat4['TIME'].dtype == np.float_:
        dat4['t'] = dat4['TIME']
    else:
        try:
            dat4['t'] = pd.to_numeric(pd.to_timedelta(dat4['TIME']))/1e9/60/60/24/365.25 + 1970
        except ValueError:
            print('Warning: failed to parse TIME column, trying as float')
            dat4['t'] = pd.to_numeric(dat4['TIME'])

    dat4['VALUEnumeric'] = pd.to_numeric(dat4['VALUE'],errors='coerce')
    dat4 = dat4[['STUDY_ID','t','event','VALUE','VALUEnumeric']]  # ordering necessary for tuple extraction later; strict naming for now
    # if events is None:
    #     events = dat4['event'].unique()

    createLbubs = 'no'  # default 'no', change according to lbubs input
    if lbubs is None:
        lbubs = np.empty([len(dat4['STUDY_ID'].unique()),2], dtype=np.float_)
        createLbubs = 'yes'
    elif type(lbubs) is np.ndarray and len(lbubs.shape) == 1 and lbubs.shape[0] == 2:
        if np.isnan(lbubs[1]):
            createLbubs = 'upper'
        if np.isnan(lbubs[0]):
            if np.isnan(lbubs[1]):
                createLbubs = 'yes'
            else:
                createLbubs = 'lower'
        lbubs = np.tile(lbubs, (len(dat4['STUDY_ID'].unique()),1))
        # print(lbubs.shape)
        # pdb.set_trace()
    elif type(lbubs) is np.ndarray and len(lbubs.shape) == 2 and lbubs.shape[1] == 2:
        pass
    else:
        print('Warning: error parsing lbubs. Auto-determining lbubs')
        lbubs = np.empty([len(dat4['STUDY_ID'].unique()),2], dtype=np.float_)
        createLbubs = 'yes'
    if createLbubs != 'no':  # modify lbubs according to createLbubs string
        # if verbose:
        #     print('Running groupby')
        dat4gp = dat4.groupby('STUDY_ID')
        if createLbubs == 'upper' or createLbubs == 'yes':
            lbubs[:,1] = dat4gp['t'].max()
        if createLbubs == 'lower' or createLbubs == 'yes':
            lbubs[:,0] = dat4gp['t'].min()
                
    # (a) subdivide data into numeric and anumeric
    # (b) groupby STUDY_ID, event and sample Tmax if > Tmax.
    # (c) collect and make a tensor
    if tvShape is None:
        tvShape = (lbubs.shape[0], 50, 2)
    else:
        if tvShape[0] is None:
            tvShape[0] = lbubs.shape[0]
        if tvShape[1] is None:
            tvShape[1] = 50
        if tvShape[2] is None:
            tvShape[2] = 2
        
    numi = ~np.isnan(dat4['VALUEnumeric'])
    pointi = np.array(dat4['VALUE'], dtype=str) == 'nan'
    dat4numeric = dat4[numi]
    dat4point = dat4[pointi]
    dat4categorical = dat4[(~numi) & (~pointi)].copy()
    # numeric and point go into pTVT
    # categorical get event mutated and go into pTVT
    dat4categorical['event'] = dat4categorical['event'] + '|' + dat4categorical['VALUE'].map(str)
    # dat4 = dat4.append(dat4categorical)

    dat4gp = dat4[numi | pointi].groupby(['STUDY_ID','event'])
    blank = np.empty(tvShape, dtype=np.float_)
    blank[:] = np.nan
    ptsTimesValuesTensor = {}  # dict['eventName'][N x Tmax x E]
    idsMap = {k:v for v,k in enumerate(dat4['STUDY_ID'].unique())}
    for groupTuple, groupDf in tqdm(dat4gp):
        pt, event = groupTuple
        # map eventdf to [N x Tmax x E]
        if groupDf.shape[0] > tvShape[1]:  # i.e. > Tmax
            sGroupDf = groupDf.sample(tvShape[1])
        else:
            sGroupDf = groupDf
        if event in ptsTimesValuesTensor:
            tensor = ptsTimesValuesTensor[event]
        else:
            tensor = blank.copy()
        tensor[idsMap[pt],0:sGroupDf.shape[0],:] = sGroupDf[['t','VALUEnumeric']]
        ptsTimesValuesTensor[event] = tensor
    # pdb.set_trace()

    # add the categorical tensors
    if events is None:
        events = np.concatenate((dat4[numi|pointi]['event'].unique(),
                                 dat4categorical['event'].unique()))
    dat4gp = dat4categorical.groupby(['STUDY_ID','event'])
    for groupTuple, groupDf in tqdm(dat4gp):
        pt, event = groupTuple
        # map eventdf to [N x Tmax x E]
        if groupDf.shape[0] > tvShape[1]:  # i.e. > Tmax
            sGroupDf = groupDf.sample(tvShape[1])
        else:
            sGroupDf = groupDf
        if event in ptsTimesValuesTensor:
            tensor = ptsTimesValuesTensor[event]
        else:
            tensor = blank.copy()
        tensor[idsMap[pt],0:sGroupDf.shape[0],:] = sGroupDf[['t','VALUEnumeric']]
        ptsTimesValuesTensor[event] = tensor

        
    # use ptsTimesValuesTensor to create waveArrayTensor
    waveArrayTensor = timelineToWaveArrayTensors(target, ptsTimesValuesTensor, bins=eventBins, ranges=ranges)

    if vectorType is Variable:
        if verbose:
            print('Converting to torch Variables')
        for event in ptsTimesValuesTensor.keys():
            ptsTimesValuesTensor[event] = Variable(torch.from_numpy(ptsTimesValuesTensor[event]), requires_grad=False)
            if event in waveArrayTensor and waveArrayTensor[event] is not None:
                if type(waveArrayTensor[event]['x']) is not Variable:
                    waveArrayTensor[event]['x'] = Variable(torch.from_numpy(waveArrayTensor[event]['x']), requires_grad=False)
                if type(waveArrayTensor[event]['wavelet']) is not Variable:
                    waveArrayTensor[event]['wavelet'] = Variable(torch.from_numpy(waveArrayTensor[event]['wavelet']), requires_grad=True)

    # pdb.set_trace()
    # censorIndexer: c x wAT['x'], censorTimes -> c x wAT-like 01 matrix for hadamard
    # i.e. per event, c: wAT['wavelet']-like.

    # high-pass regularizer:
    # Hadamard h(len(w))*(.) then reduce on wAT['wavelet']
    
    return target, events, lbubs, tvShape, ptsTimesValuesTensor, waveArrayTensor
