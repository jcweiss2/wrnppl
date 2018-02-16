
import pandas as pd
import numpy as np
import os.path
import torch
import pdb
from torch.autograd import Variable
import jcw_pandas_loader as jpl
import jcw_f2mixture as f2m
import importlib
importlib.reload(jpl)


##################
### Run ##########
##################
def default_dat4_dataset_dict():
    dataSetDict = {}
    dataSetDict['dataDir'] = '~/workspace/marshfield/extracts/'
    dataSetDict['prefix'] = 'diabetes2012fold'
    dataSetDict['suffix'] = '.csv'
    dataSetDict['folds'] = np.arange(2)+1
    dataSetDict['target'] = 'outcome|polyneuropathy'
    return dataSetDict


def load_dat4_dataset(dataSetDict=None, iterString='folds', tvShape=None, lbubs=None, eventBins=None, vectorType=Variable, verbose=False):
    ### Parameters ###
    if dataSetDict is None:
        dataSetDict = default_dat4_dataset_dict()
    # pdb.set_trace()
    
    dataFiles = [os.path.expanduser(dataSetDict['dataDir'] + dataSetDict['prefix'] + str(f) + dataSetDict['suffix']) for f in dataSetDict[iterString]]
    # print(dataFiles)

    ### Load files ###
    if verbose:
        print('Loading files')
    dats = [pd.read_csv(dataFile, dtype={'STUDY_ID':np.int64}, parse_dates=[1]) for dataFile in dataFiles]
    # for dat in dats:
    #     dat['VALUEnumeric'] = pd.to_numeric(dat.VALUE, errors='coerce')

    # fast for test
    # dats = [dat.head(30000) for dat in dats]
    
    ### Create matrices necessary for running f2mixture
    if verbose:
        print('Making data matrices')
    datObjects = np.empty(len(dats), dtype=list)
    backupTvShape = tvShape
    for i, dat in enumerate(dats):
        tvShape = backupTvShape.copy()  # arrays are passed as pointers apparently
        if (np.array([d in ['STUDY_ID','TIME','event','VALUE'] for d in dat.columns])-1).sum() == 0:  # has correct columns
            dat = dat[['STUDY_ID','TIME','event','VALUE']]
        # else:
        #     dat.columns = ['STUDY_ID','TIME','event','VALUE']  # Name arbitrarily and hope
        datObjects[i] = jpl.timelineToPricklyValues(dat,
                                                    dataSetDict['target'],
                                                    events=None,
                                                    lbubs=lbubs,
                                                    tvShape=tvShape,
                                                    eventBins=eventBins,
                                                    vectorType=vectorType,
                                                    verbose=verbose)

    # _, events, lbubs, ptsTimesValuesTensor, waveArrayTensor = datObjects[0]

    if verbose:
        print('Data loaded')

    return datObjects


def load_dat4_not_cheating_test(dataFile, outcome, Tmax=20, studyIds=None, tvShape=None, lbubs=None, eventBins=None, vectorType=Variable, ranges=None, verbose=False):
    dat = pd.read_csv(dataFile)
    dat.columns = ['STUDY_ID','TIME','event','VALUE']
    dat = dat.sort_values(['STUDY_ID','TIME']).reset_index(drop=True)
    ptsParts = []
    if studyIds is None:
        studyIds = [dat.head().ix[0,'STUDY_ID']]
    for studyId in studyIds:
        pt = dat[dat['STUDY_ID'] == studyId].groupby('event').head(Tmax).reset_index(drop=True).copy()
        outcomeis = np.where(pt['event'] == outcome)[0]
        if len(outcomeis) > 0:
            outcomeis = outcomeis[outcomeis > 0]
        
        for i, outcomei in np.ndenumerate(np.flipud(outcomeis)):
            ptPart = pt[:outcomei].copy()
            ptPart['STUDY_ID'] = ptPart['STUDY_ID'] + i + 1
            ptsParts.append(ptPart)
    dataSet = pt.append(ptsParts)
    # pdb.set_trace()
    if tvShape is not None:
        tvShape = list(tvShape).copy()
    datObject = jpl.timelineToPricklyValues(dataSet,
                                            outcome,
                                            events=None,
                                            lbubs=lbubs,
                                            tvShape=tvShape,
                                            eventBins=eventBins,
                                            vectorType=vectorType,
                                            ranges=ranges,
                                            verbose=verbose)
    return datObject, dataSet
