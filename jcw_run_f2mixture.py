
# coding: utf-8

# In[1]:


import importlib
import matplotlib.pyplot as plt
import jcw_pywavelets
import jcw_f2mixture as f2m
import jcw_run_dat4 as jrd
import jcw_utils
import torch
import numpy as np
import pandas as pd
from torch.autograd import Variable
from torch import nn
import pdb
import os
import datetime as dt
import jcw_hawkes as jh
import argparse
importlib.reload(jrd)
importlib.reload(f2m)
importlib.reload(jh)

#############################
############ RUN ############
#############################

if __name__ == '__main__':

    # Setup `argparse` module to assist in parsing the command line arguments.

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_directory", type=str, default='~/workspace/lt/py/data/targetHgbA1C/')
    parser.add_argument("--source_filename", type=str, default='a1ctarget4.csv')
    parser.add_argument("--N", type=int, default=100, help="Sample Size")
    parser.add_argument("--E", type=int, default=2, help="Event Dimension (time counts)")
    parser.add_argument("--D", type=int, default=2, help="# of reduce functions")
    parser.add_argument("--Ebins", type=int, default=32, help="")
    parser.add_argument("--scaleSteps", type=int, default=16, help="")
    parser.add_argument("--Tmax", type=int, default=50, help="Max. # of target times to use in tensor")
    parser.add_argument("--targetSteps", type=int, default=80, help="Target Steps")
    parser.add_argument("--batch", type=int, default=10, help="Batch")
    parser.add_argument("--steps", type=int, default=5, help="Initial Run Length (below is much longer)")
    parser.add_argument("--minHazard", type=int, default=5, help="")
    parser.add_argument("--maxGainOnHit", type=float, default=np.log(10), help="")
    parser.add_argument("--violationMultiplier", type=int, default=100, help="")
    parser.add_argument("--censorSize", type=int, default=1, help="overriden below if desired censorVector")
    parser.add_argument("--npermutations", type=int, default=4, help="# of permutations")
    parser.add_argument("--imageDir", type=str, default="./images/", help="Output Directory")
    parser.add_argument("--isCensorSizeDefault", type=bool, default=True, help="Output Directory")
    parser.add_argument("--pL1Value", type=float, default=1e-7, help="L1 regularization parameter")
    parser.add_argument("--pL2Value", type=float, default=0, help="L2 regularization parameter")
    parser.add_argument("--doingHawkes", type=bool, default=False, help="True if doing Hawkes")
    parser.add_argument("--optim", type=str, default="RMSprop", help="which optimizer to use -Adam/RMSprop?")
    parser.add_argument("--batchisUserInput", type=int, default=-999, help="type a list of user id's for the user you need the graph")

    args = parser.parse_args()



    # Allocate the command line parameters to global variables of this script.

    data_directory = args.data_directory
    source_filename_prefix = '.'.join(args.source_filename.split(".")[:-1])
    source_filename_suffix = '.' + args.source_filename.split(".")[-1]

    N = args.N
    E = args.E
    D = args.D
    Ebins = args.Ebins
    scaleSteps = args.scaleSteps
    Tmax = args.Tmax
    targetSteps = args.targetSteps
    steps = args.steps
    batch = args.batch
    minHazard = args.minHazard
    maxGainOnHit = args.maxGainOnHit
    violationMultiplier = args.violationMultiplier
    censorSize = args.censorSize
    npermutations = args.npermutations
    imageDir = args.imageDir
    pL1Value=args.pL1Value
    pL2Value=args.pL2Value
    doingHawkes=args.doingHawkes
    optim=args.optim
    isCensorSizeDefault=args.isCensorSizeDefault
    batchisUserInput=args.batchisUserInput


np.random.seed(np.int(1.5829e5))  # 1.5829e5
torch.manual_seed(np.int(1.5829e5+252))  # 252

# N = 100  # 40 # samples
# E, D = 2, 2  # eventDimension (time counts), reduceFunctions
# Ebins = 32
# scaleSteps = 16
# Tmax = 20  # max number of target times to use in tensor
# targetSteps = 80  # 200
# batch = 10
# steps = 5
# minHazard = 1e-5
# maxGainOnHit = np.log(10)
# violationMultiplier = 100
# censorSize = 1  # default, overridden below if desired censorVector
# pL1Value=1e-7
# pL2Value=0
# doingHawkes=False
# optim="RMSprop"
# isCensorSizeDefault=True
# batchisUserInput=-999

if(isCensorSizeDefault):
    censorSize=censorSize
else:
    censorVector = Variable(torch.linspace(start=-1,end=-0,steps=3).double(),requires_grad=False)
    if 'censorVector' in globals():
        censorSize = censorVector.size()[0]
    censorFixed = False

# npermutations = 4

# imageDir = 'images/'  # for output
timeString = str(dt.datetime.now())

tensorType = torch.DoubleTensor
# tensorType = torch.cuda.DoubleTensor

pL1 = Variable(torch.DoubleTensor([pL1Value]).type(tensorType),requires_grad=False)  # parameter for L1 reg.
pL2 = Variable(torch.DoubleTensor([pL2Value]).type(tensorType),requires_grad=False)  # parameter for L2 reg.
# pL2 = Variable(torch.DoubleTensor([1e-4]).type(tensorType),requires_grad=False)  # parameter for L2 reg.

hd = jcw_pywavelets.create_haar_dictionary(10)
for i in range(len(hd)):
    if tensorType == torch.cuda.DoubleTensor:
        hd[i] = Variable(torch.from_numpy(np.array(hd[i])).type(torch.cuda.DoubleTensor), requires_grad=False)
    else:
        hd[i] = Variable(torch.from_numpy(np.array(hd[i])).type(torch.DoubleTensor), requires_grad=False)

### Simulation data
# trainDir = 'data/targetKetoacidosis/train/'
# testDir = 'data/targetKetoacidosis/test/'
# trainDir = 'data/targetHgbA1C/train/'
# testDir = 'data/targetHgbA1C/test/'
# # target, events, lbub, ptsTimesValuesTensor, waveArrayTensor = f2m.loadPricklyData(trainDir, (N, Tmax, E))
# # # _, _, testLbub, testPtsTimesValuesTensor, testWaveArrayTensor = f2m.loadPricklyData(testDir, (N, Tmax, E))

details = {'dataDir': data_directory,
           'prefix': source_filename_prefix,
           'suffix':source_filename_suffix,
           'folds': ['','test'],
           'target': 'HgbA1C',
           'holdout': ['holdout']}
# details = {'dataDir': '~/workspace/lt/py/data/targetHgbA1C/',
#            'prefix': 'a1ctarget4',
#            'suffix':'.csv',
#            'folds': ['','test'],
#            'target': 'Ketoacidosis',
#            'holdout':['holdout']}
# details = {'dataDir': '~/workspace/lt/py/data/',
#            'prefix': 'tnl1e4Data',
#            'suffix':'.csv',
#            'folds': ['','Test'],
#            'target': 'ACS',
#            'holdout': ['Holdout']}

# # (TEST not cheating) #
# target, events, lbub, trainParams, ptsTimesValuesTensor, waveArrayTensor = \
#     jrd.load_dat4_dataset(dataSetDict=details, tvShape=np.array([None, Tmax, None]), eventBins=64)[0]
# testStuff = jrd.load_dat4_not_cheating_test('data/targetHgbA1C/a1ctarget4test.csv', 'HgbA1C', Tmax=Tmax, tvShape=[None, Tmax, E])
# _, _, testLbub, testParams, testPtsTimesValuesTensor, testWaveArrayTensor = testStuff[0]
# testData = testStuff[1]
# # RUN train and test #
# # (end TEST) #

datObjects = jrd.load_dat4_dataset(dataSetDict=details, tvShape=np.array([None, Tmax, None]), eventBins=Ebins)
target, events, lbub, trainParams, ptsTimesValuesTensor, waveArrayTensor = datObjects[0]
_, _, testLbub, testParams, testPtsTimesValuesTensor, _ = datObjects[1]

N = trainParams[0]
Ntest = testParams[0]

pL1Wvt = Variable(torch.DoubleTensor([1./N]).type(tensorType),requires_grad=False)  # parameter for L1 reg. 1-d images
pL1WvtImage = Variable(torch.DoubleTensor([1./N]).type(tensorType),requires_grad=False)  # parameter for L1 reg. 2-d images
# pL1Wvt = Variable(torch.DoubleTensor([0.1/N]).type(tensorType),requires_grad=False)  # parameter for L1 reg. 1-d images
# pL1WvtImage = Variable(torch.DoubleTensor([0.1/N]).type(tensorType),requires_grad=False)  # parameter for L1 reg. 2-d images

hawkesrd, hawkespd = None, None  # leave uncommented despite not using
if(doingHawkes):
    print('Doing hawkes')
    hawkesrd, hawkespd = jh.makeHawkesReconstructionDict(waveArrayTensor)
    E, D =  1, 1

### build the derivedTensors
wsizes = f2m.getWaveletSizes(events, waveArrayTensor)
stepsValues, stepsTimes, w2ds, valueBinIndices, dim2d =     f2m.getImageTensors(target, events, wsizes, ptsTimesValuesTensor, scaleSteps)

watx, countsInWatxTensor, distsInWatxTensor = f2m.derivedTensors(lbub, target, ptsTimesValuesTensor, targetSteps)
ssmaps = f2m.getSSMaps(ptsTimesValuesTensor, waveArrayTensor, watx)

f2m.attachPenalties(waveArrayTensor)  # 1d penalties
p2ds = f2m.getPenalties2d(w2ds)  # 2d penalties
coefficientPenalties = {'1d': 'coefficientPenalty', '2d': p2ds}  # penalty dictionary
if 'censorVector' in globals() and censorVector is not None:
    hadamardCensor = f2m.makeHadamardCensor(censorVector, waveArrayTensor)
else:
    hadamardCensor = None

testWatx, testCountsInWatxTensor, testDistsInWatxTensor = f2m.derivedTensors(testLbub, target, testPtsTimesValuesTensor, targetSteps)
# tW = len(testWatx)-1
# for i in range(0, len(testWatx)-1):
#     testWatx[i] = testWatx[tW]
#     testCountsInWatxTensor[i] = testCountsInWatxTensor[tW]
#     testDistsInWatxTensor[i] = testDistsInWatxTensor[tW]
# testWatx[9] = testWatx[8]; testCountsInWatxTensor[9] = testCountsInWatxTensor[8]; testDistsInWatxTensor[9] = testDistsInWatxTensor[8]  # Hack 9 to have 8 counts
testSsmaps = f2m.getSSMaps(testPtsTimesValuesTensor, waveArrayTensor, testWatx)

# CUDA declarations
wAToff = waveArrayTensor  # outside to include in pars
woff = w2ds  # same
if tensorType == torch.cuda.DoubleTensor:  # make cuda (if desired) out of existing tensors
    pTVToff = ptsTimesValuesTensor
    for i, pti in enumerate(ptsTimesValuesTensor):
        ptsTimesValuesTensor[pti] = ptsTimesValuesTensor[pti].cuda()
    for i, wat in enumerate(waveArrayTensor):
        for j, watpart in enumerate(waveArrayTensor[wat]):
            # let's redeclare instead using the Variable(.cuda()) trick
            if watpart == 'wavelet':
                waveArrayTensor[wat][watpart] = Variable(waveArrayTensor[wat][watpart].data.cuda(), requires_grad=True)
            else:
                waveArrayTensor[wat][watpart] = waveArrayTensor[wat][watpart].cuda()
    cIWToff = countsInWatxTensor
    dIWToff = distsInWatxTensor
    for i, ci in enumerate(countsInWatxTensor):
        countsInWatxTensor[i] = countsInWatxTensor[i].cuda()
        distsInWatxTensor[i] = distsInWatxTensor[i].cuda()
    sMoff = ssmaps
    for i, ei in enumerate(ssmaps):
        for j, ji in enumerate(ssmaps[ei]):
            ssmaps[ei][ji] = ssmaps[ei][ji].cuda()
    sVoff = stepsValues
    sToff = stepsTimes
    for i, svi in enumerate(stepsValues):
        stepsValues[i] = stepsValues[i].cuda()
        stepsTimes[i] = stepsTimes[i].cuda()
        # redeclare
        w2ds[i] = Variable(w2ds[i].data.cuda(), requires_grad=True)
        

class ParallelLinear(nn.Module):
    '''
    Acts as (dim) c linear layers. Set fixedLayers=True to share parameters across c 
    '''
    def __init__(self, isize, osize, dimSize, dim, fixedLayers=False, collapseDims=True):
        super(ParallelLinear,self).__init__()
        if fixedLayers:
            self.ll = nn.ModuleList([nn.Linear(isize,osize)])
        else:
            self.ll = nn.ModuleList([nn.Linear(isize, osize) for i in range(dimSize)])
        self.fixedLayers = fixedLayers
        self.overridden = False
        self.dim = dim
        self.weight = torch.cat([ll.weight for ll in self.ll],0)
        self.bias = torch.cat([ll.bias for ll in self.ll])
        self.collapseDims = collapseDims

    def overrideLl(self, llList):
        '''
        If you want a more complicated mapping from isize -> osize, create your ll and use here right after initialization.
        llList is a list applied in parallel (across c)
        '''
        if hasattr(self, 'usedForward') and self.usedForward == True:
            print('Cannot call overrideLl. forward(.) already called')
            return
        self.overridden = True
        self.ll = nn.ModuleList(llList)
        return
        
    def forward(self, x):  # x: len(w) x (C) x rxd x E
        self.usedForward = True
        dims = x.size()
        if self.collapseDims:  # use True with vanilla linear layer, but modify if you need the dims for custom ll
            # pdb.set_trace()
            x = torch.cat(x.split(1,2),3).squeeze(2)  # len(w) x (C) x [rxd x E]
            # x = x.contiguous().view(dims[0],dims[1],-1)  # len(w) x (C) x [rxd x E]
        ds = x.split(1,dim=self.dim)  # collapse rxd and E
        if self.fixedLayers:
            out = torch.cat([self.ll[0](d) for d in ds], dim=self.dim)
        else:
            # pdb.set_trace()
            out = torch.cat([ll(d) for d, ll in zip(ds, self.ll)], dim=self.dim)
        return out

# isize (RxDxE) -> osize
# isize -> batchNorm + permuteLayers + maxPoolLayers


class NormPermutePool(nn.Module):
    '''
    BatchNorm over reduce x image_dims, permute and pool over _ x (C) x [R x D] x E 
    '''
    def __init__(self, isize, rxd, osize, permutations=1):
        super(NormPermutePool,self).__init__()
        self.bn = nn.BatchNorm1d(rxd, affine=False)
        self.rxd = rxd
        self.E = int(isize/rxd)
        self.permuteTensorShape = (rxd, isize/rxd)
        self.osize = osize
        self.permutations = permutations
        self.rxdpermutation = np.random.permutation(rxd)
        self.signTensor = Variable(((torch.rand(self.E,self.E,permutations)>0.5).double()*2-1))
        if rxd > 1:
            while np.all(self.rxdpermutation == np.arange(rxd)):
                self.rxdpermutation = np.random.permutation(rxd)  # generate non-identical permutation, but ...
        # pdb.set_trace()
        self.epermutation = np.random.permutation(self.E)
        if self.E > 1:
            while np.all(self.epermutation == np.arange(self.E)):
                self.epermutation = np.random.permutation(self.E)  # field could be e.g. mod 2 so permutations could be identical
        self.permuteRxdMatrix = Variable(torch.eye(rxd).double())[:,self.rxdpermutation]  # rxd x rxd
        permuteEMatrix = Variable(torch.eye(self.E).double())[:,self.epermutation]  # E x E
        permuteEList = []
        permuter = permuteEMatrix.clone()  # E x E
        for i in range(permutations):
            permuteEList.append(permuter)
            permuter = permuter.matmul(permuteEMatrix)
        permuteEList = [p.unsqueeze(2) for p in permuteEList]
        self.permuteETensor = torch.cat(permuteEList, dim=2)  # E x E x permutations
        kernelSizes = np.power(2,np.arange(permutations)).tolist()
        self.pools = [nn.MaxPool1d(kernel_size=k, stride=k, ceil_mode=True, return_indices=True) for k in kernelSizes]
        self.unpools = [nn.MaxUnpool1d(kernel_size=k, stride=k) for k in kernelSizes]
        self.ll = nn.Linear(rxd*self.E*permutations, osize).double()
        
        self.soft = nn.Softmax().double()
        self.weight = self.ll.weight
        self.bias = self.ll.bias
        self.mlist = nn.ModuleList([self.ll])
        
    def forward(self, x):
        '''
        Expects data of size _ x (C) x rxd x E
        '''
        dims = x.size()
        rdim, edim = len(dims)-2, len(dims)-1
        # pdb.set_trace()
        # out = self.bn(x.contiguous().view(-1, self.rxd, self.E)).view(x.size()[0], x.size()[1], self.rxd, self.E)  # _ x (C) x rxd x E
        out = x.transpose(rdim,edim).matmul(self.permuteRxdMatrix)  # _ x (C) x E x rxd
        out = out.transpose(rdim,edim).matmul(             (self.permuteETensor*self.signTensor).view(self.E,-1))  # _ x (C) x rxd x [E x permutations]
        out = out.view(dims[0],dims[1],dims[2],self.E,-1)  # _ x (C) x rxd x E x permutations
        outi = []
        for p, u, o in zip(self.pools, self.unpools, out.split(1,dim=4)):
            pool, indices = p(o.contiguous().view(x.size()[0], x.size()[1], -1))
            outi.append(u(pool,indices))  # [_ x (C) x (rxd x E)]
            
        outi[len(outi)-1] = outi[len(outi)-1][:,:,:outi[0].size()[2]]  # make shapes match for pool erraticity # TODO fix hack # fails for Hawkes?

        out = torch.cat(outi,dim=2)
        # pdb.set_trace()
        out = self.ll(out)  # _ x (C) x osize
        return out 


print('Running steps')
baseRate = (~np.isnan(ptsTimesValuesTensor[target].data.numpy())).sum() / (lbub[:,1]-lbub[:,0]).sum()
parallelLinearLayer = ParallelLinear(E*D*len(events), 1, censorSize, dim=1).double()  # Use for linear layer only
npermutations = 1
# parallelLinearLayer = ParallelLinear(E*D*len(events), 1, censorSize, dim=1, collapseDims=False).double()
# npp = [NormPermutePool(E*D*len(events), E*D, 1, permutations=npermutations) for i in range(censorSize)]
# parallelLinearLayer.overrideLl(npp)

parallelLinearLayer.weight.data.uniform_(baseRate/(E*D*len(events)))
parallelLinearLayer.bias.data.uniform_(baseRate)
loff = parallelLinearLayer  # (cuda)
if tensorType == torch.cuda.DoubleTensor:
    parallelLinearLayer = parallelLinearLayer.cuda()
reluLayer = torch.nn.ReLU().double()
pars = []
parsWvt = [w['wavelet'] for w in wAToff.values() if w is not None]
# create regularizer multiplier for each p in pars -- based on forecast and window width
# strong penalty for violating constraint. lagrangian/KKT form?
parsWvtImage = [p for _,p in np.ndenumerate(woff) if p is not None]
parsHawkes = []
if hawkespd is not None:
    parsHawkes = [p for p in hawkespd.values()]
linearLayerPars = [p for p in loff.parameters()]
pars = parsWvt + parsWvtImage + linearLayerPars + parsHawkes

#argparse argument
if(optim=="Adam"):
    optimizer = torch.optim.Adam(pars, weight_decay=1e-3)
else:
    optimizer = torch.optim.RMSprop(pars, weight_decay=1e-2)

# Now use waveArrayTensor and ptsTimesValuesTensor for learning
    
    
def runStep(t, N, Tmax, targetSteps, optimizer, events, ptsTimesValuesTensor, waveArrayTensor, hd, ssmaps, countsInWatxTensor, distsInWatxTensor, w2ds, valueBinIndices, indexBatch=None, verbose=0, hadamardCensor=None, coefficientPenalties=None, withpdb=False, overrideReconstructionDict=None, overrideParametersDict=None):
    if verbose:
        print('step ' + str(t))
    if indexBatch is None:
        indexBatch = np.arange(N)

    B = len(indexBatch)
    C = 1
    if hadamardCensor is not None:
        C = hadamardCensor['shape0']  # hC is C x len(w), TODO make string 'shape0' protected
        blank = Variable(torch.zeros((1, C, targetSteps)).double().type(tensorType), requires_grad=False)
    else:
        blank = Variable(torch.zeros((1, targetSteps)).double().type(tensorType),
                         requires_grad=False)
    blankTmax = np.empty(Tmax, dtype=Variable)  # fix to match hadamard
    for i in range(len(blankTmax)):
        blankTmax[i] = blank
    interpolated = np.empty(B, dtype=np.ndarray)
    imageInterpolated = np.empty(B, dtype=np.ndarray)
    # interpolated = Variable(torch.DoubleTensor(B,len(events),Tmax,targetSteps), requires_grad=False)
    #   while this is conceptually nicer, python can't release memory as easily under this tensor approach
    # hitsArray = np.empty(N, dtype=Variable)
    hitsArray = Variable(torch.zeros((B,C)).type(tensorType), requires_grad=False)
    areaArray = Variable(torch.zeros((B,C)).type(tensorType), requires_grad=False)  # np.empty(N, dtype=Variable)
    lcLayerArray = Variable(torch.zeros((B, C, targetSteps)).type(tensorType), requires_grad=False)
    rlLayerArray = Variable(torch.zeros((B, C, targetSteps)).type(tensorType), requires_grad=False)
    violationArray = Variable(torch.zeros((B, C)).type(tensorType), requires_grad=False)

    # reconstruct the images for mapping onto hazard
    reconstruction = np.empty(len(events), dtype=Variable)
    images = np.empty(len(events), dtype=Variable)
    if overrideReconstructionDict is None:
        for eventi, event in enumerate(events):
            if event not in waveArrayTensor or waveArrayTensor[event] is None:
                continue  # skey will also be none so np.empty will not matter
            powwat = waveArrayTensor[event]['wavelet'].size()[0]
            wat = int(np.log2(powwat))
            reconstruction[eventi] = waveArrayTensor[event]['wavelet'].unsqueeze(0).mm(hd[wat-1])  # reconstruct: 1xL x LxL
        for eventi, event in enumerate(events):
            if w2ds[eventi] is None:
                continue  # skey will also be none so np.empty will not matter
            if w2ds[eventi].size() == (1, 1):
                images[eventi] = Variable(torch.zeros(int(dim2d[eventi,0]),int(dim2d[eventi,1])).double().type(tensorType),
                                          requires_grad=False)  # break the chain; no wavelet here
            else:
                images[eventi] =                     w2ds[eventi].matmul(hd[np.log2(dim2d[eventi,1])-1]).t().                    matmul(hd[np.log2(dim2d[eventi,0])-1]).t()  # events x valueBins x timeBins
    else:
        '''
        overrideReconstructionDict is a dict with keys events and values Variables with requires_grad=True upstream.
        It is meant to allow for Hawkes approximations. You'll need to overwrite the Variable data each step but pass the gradient
        to the Hawkes parameters. To make Hawkes, you need to disable all but the additive reduceLayers (this is done is oRD is not
        None.
        '''
        for eventi, event in enumerate(events):
            if event in overrideReconstructionDict:
                reconstruction[eventi] = overrideReconstructionDict[event].unsqueeze(0)  # 1 x len(w)
                
    if optimizer is not None:
        optimizer.zero_grad()
        if withpdb:
            pdb.set_trace()

    eventsLayerB = np.empty(len(indexBatch),Variable)
    for bi, ni in enumerate(indexBatch):  # (bi=batchi, i=batchValue)
        # print 'pt ' + str(i)
        interpolationLayer = np.empty(len(events), dtype=Variable)
        imageInterpolationLayer = np.empty(len(events), dtype=Variable)
        reduceLayer = np.empty(len(events), dtype=Variable)

        interpolated[bi] = np.empty(len(events), dtype=np.ndarray)
        imageInterpolated[bi] = np.empty(len(events), dtype=np.ndarray)

        # Get hazard functions per event
        for eventi, event in enumerate(events):
            # TODO waiting, sparseTensor is only implemented for matrices (2-D), and not even
            # jcw_helper.dot(ssmaps[event],applied)  #NotImplementedError, expand not available
            # appliedOnTimes = ssmaps[event].matmul(applied)  # NotImplementedError, bmm not available, have to do it manually
            interpolated[bi][eventi] = blankTmax.copy()  # COPY indices (shallow) np.empty(Tmax, dtype=Variable)
            imageInterpolated[bi][eventi] = blankTmax.copy()  # COPY (shallow)            
            for ti in np.arange(Tmax):
                skey = None
                if event in ssmaps:
                    skey = f2m.getSSMap(ni, ti, ssmaps[event], Tmax)
                # break
                if skey is not None:
                    recon = reconstruction[eventi]
                    if hadamardCensor is not None:
                        reconCensored = recon.expand_as(hadamardCensor[event]) *                             hadamardCensor[event]  # C x len(w)
                        recon = reconCensored.unsqueeze(0)  # 1 x C x len(w)

                    # skey len(w) x ts; recon.matmul(skey) is C x ts  # note: matmul() broadcasts, mm() does not
                    interpolated[bi][eventi][ti] = recon.matmul(skey)
                    if valueBinIndices[ni][eventi] is None or len(valueBinIndices[ni][eventi].shape) == 0:
                        imageInterpolated[bi][eventi][ti] = blank  # fix blank tensor size
                    else:
                        if ti >= len(valueBinIndices[ni][eventi]):
                            imageInterpolated[bi][eventi][ti] = blank
                        elif overrideReconstructionDict is not None:
                            continue
                        else:
                            imageRecon = images[eventi][valueBinIndices[ni][eventi][ti],:].unsqueeze(1).t()
                            if hadamardCensor is not None:
                                imageCensored = imageRecon.expand_as(hadamardCensor[event]) *                                     hadamardCensor[event]  # C x len(w)
                                imageRecon = imageCensored.unsqueeze(0)  # 1 x C x len(w)
                            # imageInterpolated[bi][eventi][j] = skey.t().matmul(  # TODO make it able to hadamard and use imageCensored
                            #     imageRecon.t()).t()  # which row!?
                            imageInterpolated[bi][eventi][ti] = imageRecon.matmul(skey)
                            
                        # TODO are you flipping the image appropriately? Easy flipping not implemented
                        # So when constructing, construct appropriately;
                        # just need to flip times
                        #   (values are ordered low (top of matrix) to high (bottom) appropriately)
                        # skey is flipped actually;
                else:
                    break
                
            # interpolationLayer[eventi] = torch.cat(interpolated[bi],0)
            # pdb.set_trace()
            interpolationLayer[eventi] = torch.cat(
                tuple(interpolated[bi][eventi]), 0)  # Tmax x (C) x len(w)
            # interpolationLayer[eventi] = torch.cat(interpolated[bi][eventi],0)
            sumLayer = interpolationLayer[eventi].sum(0,keepdim=True)
            maxLayer = interpolationLayer[eventi].max(0,keepdim=True)[0]  # CHECK [0]?
            # interpolationLayer[eventi] = interpolated[bi,eventi,:,:] #only one object can store
            # sumLayer = interpolated[bi,eventi,:,:].sum(0,keepdim=True)
            # maxLayer = interpolated[bi,eventi,:,:].max(0,keepdim=True)[0]
            imageInterpolationLayer[eventi] = torch.cat(
                tuple(imageInterpolated[bi][eventi]), 0)  # Tmax x (C) x len(w)
            imageSumLayer = imageInterpolationLayer[eventi].sum(0,keepdim=True)  # 1 x (C) x len(w)
            imageMaxLayer = imageInterpolationLayer[eventi].sum(0,keepdim=True)  # 1 x (C) x len(w)

            if overrideReconstructionDict is None:
                reduceLayer[eventi] = torch.cat((sumLayer, maxLayer,imageSumLayer, imageMaxLayer), 0).unsqueeze(0)  # 1 x 4 x (C) x len(w)
            else:
                reduceLayer[eventi] = sumLayer.unsqueeze(0)
                
        # compute loss on pt
        eventsLayer = torch.cat(tuple(reduceLayer),0)  # E(vents) x 4 x (C) x len(w)
        eventsLayerB[bi] = eventsLayer  # for debugging
        
        # Note that Py3.0 as of 1/1/18 list comprehensions use global variables over method variables
        # if withpdb:
        #     pdb.set_trace()

        if len(eventsLayer.size()) == 3:
            eventsLayer = eventsLayer.unsqueeze(2)
        clcLayerTensor = parallelLinearLayer(eventsLayer.permute(3,2,1,0)).transpose(0,1)  # (C) x len(w) x 1
        crlLayerTensor = reluLayer(clcLayerTensor - minHazard) + minHazard  # (C) x len(w) x 1
        
        violationArray[bi,:] = torch.clamp(clcLayerTensor,max=0).abs().squeeze(2).matmul(distsInWatxTensor[ni]) *             violationMultiplier 
        
        lcLayerArray[bi,:,:] = clcLayerTensor.squeeze(2)  # (C) x len(w)
        rlLayerArray[bi,:,:] = crlLayerTensor.squeeze(2)  # (C) x len(w)

        # pdb.set_trace()
        # Note: to avoid backward(.) errors, DO NOT USE indexed subset on R.H.S., e.g. ... = rlLayerArray[bi,:,:]
        hitsArray[bi,:] = torch.clamp(torch.log(crlLayerTensor),max=maxGainOnHit).squeeze(2).matmul(countsInWatxTensor[ni])  # (C) # TODO Possibly need squeeze back when C!=1
        areaArray[bi,:] = crlLayerTensor.squeeze(2).matmul(distsInWatxTensor[ni])  # (C)  # TODO Possibly need squeeze back when C!=1
        # areaArray[bi,:] = rlLayer.t().matmul(distsInWatxTensor[i]).squeeze()  # (C)
        # ll = torch.cat(hitsArray) - torch.cat(areaArray)
        # print ll
        # loss = ll.sum().neg_()  # per individual

        # if verbose == 1:
        #     print eventsLayer.data
        # pdb.set_trace()
    l1Linear = torch.cat([p for n, p in parallelLinearLayer.named_parameters() if n.endswith('weight')], 0)
    # l1Wvt = torch.cat([v.unsqueeze(0).t() for v in pars if len(v.data.size()) == 1][:-1])  # bias term last, remove
    if coefficientPenalties is not None:
        wvtPenaltyArray = np.full(len(waveArrayTensor), Variable(torch.DoubleTensor(1)))
        wvtImagePenaltyArray = np.full(len(w2ds), Variable(torch.DoubleTensor(1,1)))
        for i, e in enumerate(waveArrayTensor.keys()):
            if waveArrayTensor[e] is not None:
                wvtPenaltyArray[i] = waveArrayTensor[e]['wavelet'] * waveArrayTensor[e][coefficientPenalties['1d']]
        for i, wi in enumerate(w2ds):
            if wi is not None:
                wvtImagePenaltyArray[i] = wi*coefficientPenalties['2d'][i]
        l1Wvt = torch.cat(tuple(wvtPenaltyArray))
        l1WvtImage = torch.cat([ wipa.view(-1) for wipa in wvtImagePenaltyArray],0) 
    else:
        l1Wvt = torch.cat(parsWvt)
        l1WvtImage = torch.cat([p.view(-1) for p in parsWvtImage],0)
    
    penalty = pL1 * l1Linear.abs().sum() + pL2 * l1Linear.pow(2).sum() +               pL1Wvt * l1Wvt.abs().sum() + pL1WvtImage * l1WvtImage.abs().sum()
    loss = (hitsArray - areaArray - violationArray).sum().neg_()/B/C + penalty
    
    if optimizer is not None:
        # pdb.set_trace()
        loss.backward()
        optimizer.step()
        if overrideReconstructionDict is not None:
            jh.resetHawkesTensors(overrideReconstructionDict, overrideParametersDict, waveArrayTensor)

    if verbose:
        print(np.round(torch.cat([hitsArray[bi,0].data,  # one example good part
                                  areaArray[bi,0].data,  # one example bad part
                                  loss.data,  # batch average with regularization
                                  ((hitsArray - areaArray).sum().neg_()/B/C).data,  # batch average without regularization component
                                  hitsArray.mean().data/C  # batch average good part
                                  ]).cpu().numpy(), 3))
    # break
    if(withpdb):
        pdb.set_trace()
    return loss, hitsArray, areaArray, lcLayerArray, rlLayerArray, ptsTimesValuesTensor[target][:, :, 0]


#########Start of custom function###########
def getConfusionMatrix(x_predicted, y_predicted, x_actual, y_actual, threshold):
    
    #Creating the output data frame
    df = pd.DataFrame(x_predicted)
    df.columns = ['bin']
    df['value'] = 0
    df['threshold'] = threshold
    df['predictedHazard'] = y_predicted
    df['TP'] = 0
    df['TN'] = 0
    df['FP'] = 0
    df['FN'] = 0
    
    for i,val in enumerate(x_actual):
        for index,v in enumerate(x_predicted):
            if v > val:
                break
        #We get the upper index of the bin i.e. [lowerIndex, upperIndex]
        #upperIndex[i] = index #Stores the index values

        df.iloc[index,1] += 1

    TP = 0
    TN = 0
    FP = 0
    FN = 0
    
    #Column names for df
    #0: bin
    #1: value
    #2: threshold
    #3: predictedHazard
    #4: TP
    #5: TN
    #6: FP
    #7: FN
    for index,row in df.iterrows():
        if row['predictedHazard'] >= threshold and row['value'] > 0:
            TP += 1
            df.iloc[index,4] = 1
        elif row['predictedHazard'] >= threshold and row['value'] == 0:
            FP += 1
            df.iloc[index,6] = 1
        elif row['predictedHazard'] < threshold and row['value'] == 0:
            TN += 1
            df.iloc[index,5] = 1
        else:
            FN += 1
            df.iloc[index,7] = 1
    
    
    #Making the cumulative columns
    df['cumulativeTP'] = 0
    df['cumulativeTN'] = 0
    df['cumulativeFP'] = 0
    df['cumulativeFN'] = 0
    df['cumulativeTP'] = np.cumsum(df['TP'])
    df['cumulativeTN'] = np.cumsum(df['TN'])
    df['cumulativeFP'] = np.cumsum(df['FP'])
    df['cumulativeFN'] = np.cumsum(df['FN'])
    
    #Caclulating TPR/FPR ratio
    df['TPR'] = df['cumulativeTP']/(df['cumulativeTP']+df['cumulativeFN']) #sensitivity
    df['TNR'] = df['cumulativeTN']/(df['cumulativeTN']+df['cumulativeFP']) #specificity
    df['FPR'] = 1-df['TNR']
    df['FNR'] = 1-df['TPR']
    
    return (df,TP,TN,FP,FN)
######End of custom function#######

#########Start of custom function###########
def createGraphs(df):
    #Removing rows with NaN
    df = df.dropna()
    #print(df)
    
    #Code for making 1 graph
    #Making graph for TPR/FPR vs lambda
    fig, ax = plt.subplots()
    ax.plot(df['bin'],df['TPR']/df['FPR'],"b-") #Lambd values are basically the bin values
    #ax.plot([0,1],[0,1], "k--")
    ax.set(xlabel = "lambda",
           ylabel = "TPR/FPR",
           title = "TPR/FPR vs lambda graph");
        
    #Making graph for FNR/TNR vs lambda
    #fig, ax = plt.subplots()
    ax.plot(df['bin'],df['FNR']/df['TNR'],"g-") #Lambd values are basically the bin values
    #ax.plot([0,1],[0,1], "k--")
    ax.set(xlabel = "lambda",
          ylabel = "FNR/TNR",
          title = "FNR/TNR vs lambda graph");

    """
    #Code for making 2 graphs
    #Making graph for TPR/FPR vs lambda
    fig, ax = plt.subplots(1,2)
    ax[0].plot(df['bin'],df['TPR']/df['FPR'],"b-") #Lambd values are basically the bin values
    #ax.plot([0,1],[0,1], "k--")
    ax[0].set(xlabel = "lambda",
    ylabel = "TPR/FPR",
    title = "TPR/FPR vs lambda graph");
    ax[1].plot(df['bin'],df['FNR']/df['TNR'],"b-") #Lambd values are basically the bin values
    #ax.plot([0,1],[0,1], "k--")
    ax[1].set(xlabel = "lambda",
    ylabel = "FNR/TNR",
    title = "FNR/TNR vs lambda graph");
    """
######End of custom function#######




for t in range(steps):
    loss, hitsArray, areaArray, lcLayerArray, rlLayerArray, ttimesArray = runStep(
        t, N, Tmax, targetSteps, optimizer, events, ptsTimesValuesTensor, waveArrayTensor, hd, ssmaps, countsInWatxTensor, distsInWatxTensor, w2ds, valueBinIndices, indexBatch=np.random.choice(N, batch, False), coefficientPenalties=coefficientPenalties, hadamardCensor=hadamardCensor, overrideReconstructionDict=hawkesrd, overrideParametersDict=hawkespd, verbose=True  # , withpdb=True
    )
    
# Test set stuff
# testWatx, testCountsInWatxTensor, testDistsInWatxTensor = f2m.derivedTensors(testLbub, target, testPtsTimesValuesTensor, targetSteps)
# testSsmaps = f2m.getSSMaps(testPtsTimesValuesTensor, waveArrayTensor, testWatx)
if tensorType == torch.cuda.DoubleTensor:
    for i, ei in enumerate(testSsmaps):
        for j, ji in enumerate(testSsmaps[ei]):
            testSsmaps[ei][ji] = testSsmaps[ei][ji].cuda()
    for i, ci in enumerate(testCountsInWatxTensor):
        testCountsInWatxTensor[i] = testCountsInWatxTensor[i].cuda()
        testDistsInWatxTensor[i] = testDistsInWatxTensor[i].cuda()
_, _, _, testValueBinIndices, _ =     f2m.getImageTensors(target, events, wsizes, testPtsTimesValuesTensor, scaleSteps)
testLoss, testHitsArray, testAreaArray, testLcLayerArray, testRlLayerArray, testTtimesArray = runStep(
    t, Ntest, Tmax, targetSteps, None, events, testPtsTimesValuesTensor, waveArrayTensor, hd, testSsmaps, testCountsInWatxTensor, testDistsInWatxTensor, w2ds, testValueBinIndices, coefficientPenalties=coefficientPenalties, hadamardCensor=hadamardCensor, overrideReconstructionDict=hawkesrd, overrideParametersDict=hawkespd, verbose=True)  # use waveArrayTensor no testWaveArrayTensor because waveArrayTensor holds train set modified wavelet reconstruction parameters

# targetCleaned = 'hemoglobina1c'
# targetCleaned = 'polyneuropathy'
targetCleaned = str.split(target,'|')[len(str.split(target,'|'))-1]  # last piece after '|'

baseRate = (~np.isnan(ttimesArray.data.numpy())).sum() / (lbub[:,1]-lbub[:,0]).sum()
testBaseRate = (~np.isnan(testTtimesArray.data.numpy())).sum() / (testLbub[:,1]-testLbub[:,0]).sum()
print('Average rate: ' + str(baseRate))
print('Average rate (tune): ' + str(testBaseRate))
print('Average NLL guessing: ' +       str(baseRate*(lbub[:,1]-lbub[:,0]).sum()/lbub.shape[0] - np.log(baseRate)*baseRate*(lbub[:,1]-lbub[:,0]).sum()/lbub.shape[0]))
print('Average NLL guessing (test): ' +       str(testBaseRate*(testLbub[:,1]-testLbub[:,0]).sum()/testLbub.shape[0] - np.log(testBaseRate)*testBaseRate*(testLbub[:,1]-testLbub[:,0]).sum()/testLbub.shape[0]))


# Plot wavelet reconstruction
if not hasattr(parallelLinearLayer, 'overridden') or parallelLinearLayer.overridden == False:
    print([events[[ i % (E*D) for i in np.where(np.abs([p for p in parallelLinearLayer.parameters()][0].data.cpu().numpy()) > 0.1)[1]]]])
    # TODO fix support identification: meaningless for permuted layer
    # if E*D == 4:  # TODO fix
    #     print(pd.DataFrame([p for p in parallelLinearLayer.parameters()][0].view(-1,E*D).data.numpy(), columns=np.repeat(events,npermutations), index=['add','max','addI','maxI']))
    # else:
    #     print(pd.DataFrame([p for p in parallelLinearLayer.parameters()][0].view(-1,E*D).data.numpy(), columns=np.repeat(events,npermutations), index=['add']))

# showwat = waveArrayTensor['Ketoacidosis']
# showwat = waveArrayTensor[target]
# showwat = waveArrayTensor['Retinopathy']
# showwat = waveArrayTensor['Dizziness']
# # showwat = waveArrayTensor['SerumLDL']
# showwat = waveArrayTensor['POC']
if 'Troponin' in waveArrayTensor:
    showwat = waveArrayTensor['Troponin']
    # showwat = waveArrayTensor['ACS']
    plt.step(showwat['x'].data.cpu().numpy()[:-1],showwat['wavelet'].unsqueeze(0).mm(hd[np.log2(showwat['wavelet'].data.shape[0])-1]).squeeze().data.cpu().numpy())
    # plt.show(
    plt.savefig(imageDir + targetCleaned + timeString + 'wvtReconstruction.svg', format='svg'); plt.clf()

if 'Troponin' in events:
    ei = np.where(events == 'Troponin')[0][0]
else:
    ei = 0
if w2ds[ei] is not None:
    plt.figure(figsize=(6.5,4.5))
    plt.imshow(w2ds[ei].matmul(hd[np.log2(w2ds[ei].data.cpu().numpy().shape)[1]-1]).t().matmul(hd[np.log2(w2ds[ei].data.cpu().numpy().shape)[0]-1]).t().data.cpu().numpy(),
               extent=(stepsTimes[ei].data.cpu().numpy().min(),
                       stepsTimes[ei].data.cpu().numpy().max(),
                       stepsValues[ei].data.cpu().numpy().min(),
                       stepsValues[ei].data.cpu().numpy().max()),
               origin='lower',
               aspect='auto', cmap='BrBG',
               norm=jcw_utils.MidpointNormalize(midpoint=0.))  # cmap='gray')
    plt.yticks(np.linspace(stepsValues[ei].data.cpu().numpy().min(),stepsValues[ei].data.cpu().numpy().max(),stepsValues[ei].data.cpu().numpy().shape[0])[::3],stepsValues[ei].data.cpu().numpy()[::3].round(3))
    plt.xticks(stepsTimes[ei].data.cpu().numpy()[::10].round(0), rotation=90)
    plt.ylabel('Troponin level')
    plt.xlabel('Relative time')
    plt.colorbar()
    # plt.show()
    plt.savefig(imageDir + targetCleaned + timeString + 'wvtImage1.svg', format='svg'); plt.clf()
 
timeString = str(dt.datetime.now())
batch = 30
batchis = np.random.choice(np.minimum(N,Ntest), batch, False)
# batchis = range(1,10)
loss, hitsArray, areaArray, lcLayerArray, rlLayerArray, ttimesArray = runStep(
    t, N, Tmax, targetSteps, optimizer, events, ptsTimesValuesTensor, waveArrayTensor,
    hd, ssmaps, countsInWatxTensor, distsInWatxTensor, w2ds, valueBinIndices,
    indexBatch=batchis,
    hadamardCensor=hadamardCensor, overrideReconstructionDict=hawkesrd, overrideParametersDict=hawkespd, 
    verbose=True)
mycolors = np.linspace(0.8,0, batch)
plt.figure(figsize=(6,4))

results = [] #Contains the final (TP,TN,FP,FN) result tuples



#Sample custom input
#batchisUserInput = [3, 4, 5, 10, 16]
batchisUserInput = -999
if batchisUserInput != -999:
    batchis = batchisUserInput #Get overwritten with the custom input
    print("Overwritting batchis with custom user input:")

if(batchisUserInput!=-999):
    for bi, ni in enumerate(batchis):
        plt.figure(bi)
        
        x_predicted = watx[ni][:-1]
        y_predicted = rlLayerArray.data.cpu().numpy()[bi,0,:].squeeze()
        y_predicted_cum = np.cumsum(y_predicted)
        
        
        if sum(~np.isnan(ttimesArray.data.cpu().numpy()[ni,:])) > 0:
            # print np.nanmin(rlLayerArray[i,:].data.cpu().numpy())
            activeis = np.where(~np.isnan(ttimesArray.data.cpu().numpy()[ni,:]))[0]
            inds = np.digitize(ttimesArray.data.cpu().numpy()[ni,:][activeis],
                               watx[ni][:-1]) - 1
                
        x_actual = ttimesArray.data.cpu().numpy()[ni,activeis]
        y_actual = np.zeros(len(activeis))+rlLayerArray.data.cpu().numpy()[bi,0,inds]
        y_actual_cum = np.cumsum(y_actual)
        
        #Plotting the cumulative hazard rate w.r.t. lambda
        
        plt.plot(x_predicted, np.cumsum(y_predicted))
        plt.xlabel('Lambda')
        plt.ylabel('Cumulative hazard rate')
        #Plotting the total hazard rate of all the predicted points at the final time
        #print('Plotting the total hazard rate of all the predicted points at the final time')
        
        plt.plot(x_predicted[-1], sum(y_actual), marker='o', markersize=5, color="red")
        
        (df,TP,TN,FP,FN) = getConfusionMatrix(x_predicted, y_predicted, x_actual, y_actual,0.1)
        results.append((df,TP,TN,FP,FN))
        
        #Plotting the graphs
        #print('Plotting the TPR/FPR and FNR/... graphs')
        plt.figure(bi+len(batchisUserInput))
        createGraphs(df)

    plt.show()
    plt.savefig(imageDir + targetCleaned + timeString + 'testHazards_Shikha' + str(c) + '.svg', format="svg"); plt.clf()

for bi, ni in enumerate(batchis):
    # plt.plot(np.linspace(0,20,num=targetSteps),testRlLayerArray.data.cpu().numpy()[i,:].squeeze())
    plt.step(watx[ni][:-1], rlLayerArray.data.cpu().numpy()[bi,0,:].squeeze(),color=plt.cm.viridis(mycolors[bi]), where='post')
    if sum(~np.isnan(ttimesArray.data.cpu().numpy()[ni,:])) > 0:
        # print np.nanmin(rlLayerArray[i,:].data.cpu().numpy())
        activeis = np.where(~np.isnan(ttimesArray.data.cpu().numpy()[ni,:]))[0]
        inds = np.digitize(ttimesArray.data.cpu().numpy()[ni,:][activeis],
                           watx[ni][:-1]) - 1
        plt.plot([ttimesArray.data.cpu().numpy()[ni,activeis],ttimesArray.data.cpu().numpy()[ni,activeis]],
                 [np.repeat(baseRate,len(activeis)),
                  np.zeros(len(activeis))+rlLayerArray.data.cpu().numpy()[bi,0,inds]], color=plt.cm.viridis(mycolors[bi]),
                 linestyle=':')
        plt.scatter(ttimesArray.data.cpu().numpy()[ni,activeis],
                    np.zeros(len(activeis))+rlLayerArray.data.cpu().numpy()[bi,0,inds], color=plt.cm.viridis(mycolors[bi]))



plt.xlabel('Time')
plt.ylabel('Hazard')
plt.yscale('log')
# plt.show()
plt.savefig(imageDir + targetCleaned + timeString + 'trainHazards.svg', format="svg"); plt.clf()
testLoss, testHitsArray, testAreaArray, testLcLayerArray, testRlLayerArray, testTtimesArray =     runStep(t, Ntest, Tmax, targetSteps, None, events, testPtsTimesValuesTensor,
            waveArrayTensor, hd, testSsmaps, testCountsInWatxTensor, testDistsInWatxTensor,
            w2ds, testValueBinIndices,
            indexBatch=batchis,
            coefficientPenalties=coefficientPenalties,
            hadamardCensor=hadamardCensor, overrideReconstructionDict=hawkesrd, overrideParametersDict=hawkespd, 
            # withpdb=True,  # TODO the mappin of example 2 at absolute time 5 is off. wavelet value but just for one example out of 10??
            verbose=True)
Cs = [0]
if 'censorVector' in globals():
    Cs = np.linspace(0,censorVector.size()[0]-1,num=5).astype(int)
manipulatingTestPlot = True
for c in Cs:
    plt.figure(figsize=(6,4))
    for bi, ni in enumerate(batchis):
        # if ni == 9:
        #     continue
        # plt.plot(np.linspace(0,20,num=targetSteps),testRlLayerArray.data.cpu().numpy()[i,:].squeeze())
        plt.step(testWatx[ni][1:], testRlLayerArray.data.cpu().numpy()[bi,c,:].squeeze(),color=plt.cm.viridis(mycolors[bi]), where='pre', alpha=0.5)
        if sum(~np.isnan(testTtimesArray.data.cpu().numpy()[ni,:])) > 0:
            # print np.nanmin(rlLayerArray[i,0,:].data.cpu().numpy())
            activeis = np.where(~np.isnan(testTtimesArray.data.cpu().numpy()[ni,:]))[0]
            xtimes = testTtimesArray.data.cpu().numpy()[ni,activeis]
            if manipulatingTestPlot:
                xtimes = xtimes + np.random.uniform(size=xtimes.shape[0]) - 0.5  # permuting and subsampling
                xsize = int(len(xtimes)*(np.random.uniform()<0.8))
                if xsize > 0:
                    xtimes = np.random.choice(xtimes, size=xsize)
            inds = np.digitize(xtimes,  # testTtimesArray.data.cpu().numpy()[ni,:][activeis],
                               testWatx[ni][:-1]) - 1
            plt.plot([xtimes,xtimes],
                     [np.repeat(baseRate,len(activeis)),
                      np.zeros(len(activeis))+testRlLayerArray.data.cpu().numpy()[bi,c,inds]], color=plt.cm.viridis(mycolors[bi]),
                     linestyle=':')
            plt.scatter(xtimes,
                        np.zeros(len(activeis))+testRlLayerArray.data.cpu().numpy()[bi,c,inds], color=plt.cm.viridis(mycolors[bi]))
    plt.xlabel('Time')
    plt.ylabel('Hazard')
    plt.ylim((testRlLayerArray.data.cpu().numpy().min()/2, testRlLayerArray.data.cpu().numpy().max()*2))
    plt.yscale('log')
    # plt.show()
    plt.savefig(imageDir + targetCleaned + timeString + 'testHazards' + str(c) + '.svg', format="svg"); plt.clf()

# testLoss, testHitsArray, testAreaArray, testLcLayerArray, testRlLayerArray, testTtimesArray = runStep(
#     t, N, Tmax, targetSteps, None, events, testPtsTimesValuesTensor, waveArrayTensor, hd, testSsmaps, testCountsInWatxTensor, testDistsInWatxTensor, w2ds, testValueBinIndices, verbose=True, withpdb=True)  # use waveArrayTensor no testWaveArrayTensor because waveArrayTensor holds train set modified wavelet reconstruction parameters

# plot a coefficient profile as a function of censor time
if not parallelLinearLayer.overridden:
    coefProfiles = np.empty(len(parallelLinearLayer.ll), Variable)
    for li, l in enumerate(parallelLinearLayer.ll):
        coefProfiles[li] = [p for p in l.parameters()][0]
    coefProfiles = torch.cat(tuple(coefProfiles),0)
    coefPlots = np.empty(coefProfiles.size()[1], object)
    plt.figure(figsize=(6,4))
    ax = plt.subplot(111)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.7, box.height])
    mycolors = np.linspace(0.8,0, coefProfiles.size()[1])
    linestyles = np.repeat(np.array([['-','--','-.',':']]),4,axis=0).flatten().tolist()
    cvline = [1]
    if 'censorVector' in globals() and censorVector is not None:
        cvline = censorVector.data.numpy()
        for li in np.arange(coefProfiles.size()[0]):
            coefPlots[li], = plt.plot(censorVector.data.numpy(),
                                      coefProfiles[:,li].data.numpy(),
                                      color=plt.cm.viridis(mycolors[li]),
                                      linestyle=linestyles[li])
            plt.legend(coefPlots,
                       np.array([events + ':' + s for s in ['add','max','add2','max2']]).flatten(),
                       bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        plt.ylabel('Coefficient value')
        plt.xlabel('Forecast distance')
        plt.savefig(imageDir + targetCleaned + timeString + 'coefProfiles.svg', format="svg"); plt.clf()

# Run for more steps
# optimizer = torch.optim.RMSprop(pars, weight_decay=1e-3)
# optimizer = torch.optim.Adam(pars, weight_decay=1e-3)  # 1e-4)
optimizer = torch.optim.Adam(pars, weight_decay=0)  # 1e-4)
moresteps = 10
batch = 20
testBatch = min(1000,Ntest)
updateEach = 1  # epochs
trainLosses = np.empty((int(np.ceil(moresteps*1./updateEach))))
trainLosses[:] = np.NaN
testLosses = np.copy(trainLosses)
bestPars, bestTestLoss = None, None
forecasts = 1
if 'censorVector' in globals() and censorVector is not None:
    forecasts = censorVector.size()[0]
for t in range(moresteps):
    batchis = np.random.permutation(range(N))
    batchlb = np.arange(0,N,batch)
    batchub = np.append(batchlb[1:],N)
    if batchub[-1] == batchub[-2]:
        batchlb, batchub = batchlb[:-1], batchub[:-1]
    batchLosses = np.zeros(len(batchlb))
    for iter, batchlbub in enumerate(zip(batchlb, batchub)):
        batchi = batchis[batchlbub[0]:batchlbub[1]]
        loss, hitsArray, areaArray, lcLayerArray, rlLayerArray, ttimesArray = runStep(
            t, N, Tmax, targetSteps, optimizer, events, ptsTimesValuesTensor, waveArrayTensor,
            hd, ssmaps, countsInWatxTensor, distsInWatxTensor, w2ds, valueBinIndices,
            indexBatch=batchi,
            # indexBatch=np.random.choice(N, batch, False),
            coefficientPenalties=coefficientPenalties, hadamardCensor=hadamardCensor, overrideReconstructionDict=hawkesrd, overrideParametersDict=hawkespd)
        batchLosses[iter] = (hitsArray - areaArray).sum().neg_().data[0]
    tli = int(np.floor(t*1./updateEach))
    if(t % updateEach == 0):
        print('Train performance: ')
        print(np.round(batchLosses.sum() / N / forecasts, 3))
        # loss, hitsArray, areaArray, lcLayerArray, rlLayerArray, ttimesArray = runStep(
        #     t, N, Tmax, targetSteps, optimizer, events, ptsTimesValuesTensor, waveArrayTensor,
        #     hd, ssmaps, countsInWatxTensor, distsInWatxTensor, w2ds, valueBinIndices,
        #     indexBatch=np.random.choice(N, batch, False),
        #     coefficientPenalties=coefficientPenalties,
        #     hadamardCensor=hadamardCensor, overrideReconstructionDict=hawkesrd, overrideParametersDict=hawkespd, 
        #     verbose=True)
        print('Test performance: ')
        testLoss, testHitsArray, testAreaArray, testLcLayerArray, testRlLayerArray, testTtimesArray =             runStep(t, Ntest, Tmax, targetSteps, None, events, testPtsTimesValuesTensor,
                    waveArrayTensor, hd, testSsmaps, testCountsInWatxTensor, testDistsInWatxTensor,
                    w2ds, testValueBinIndices,
                    indexBatch=np.random.choice(Ntest, testBatch, False),
                    coefficientPenalties=coefficientPenalties,
                    hadamardCensor=hadamardCensor, overrideReconstructionDict=hawkesrd, overrideParametersDict=hawkespd, 
                    verbose=True)
        # trainLosses[tli] = loss.data.cpu().numpy() / N
        # testLosses[tli] = testLoss.data.cpu().numpy() / N
        # trainLosses[tli] = ((hitsArray - areaArray).sum().neg_() / batch / forecasts).data.cpu().numpy()
        trainLosses[tli] = batchLosses.sum() / N / forecasts
        testLosses[tli] = ((testHitsArray - testAreaArray).sum().neg_() / testBatch / forecasts).data.cpu().numpy()

        if bestTestLoss is not None:
            print('Best performance ' + str(bestTestLoss))
        # Save best pars when test loss is minimized
        if bestPars is None or testLosses[tli] < bestTestLoss:
            bestPars = [p.data.clone() for p in pars]
            bestTestLoss = testLosses[tli]

trainplt, = plt.plot(trainLosses[~np.isnan(trainLosses)])
testplt, = plt.plot(testLosses[~np.isnan(testLosses)])
plt.legend([trainplt, testplt], ['Training set', 'Validation set'])
plt.yscale('log')
plt.savefig(imageDir + targetCleaned + timeString + 'learningCurves.svg', format="svg"); plt.clf()
# plt.show()

f2m.saveJcwModel([t, N, Tmax, Ebins, targetSteps, optimizer, events, ptsTimesValuesTensor, waveArrayTensor, hd, ssmaps, countsInWatxTensor, distsInWatxTensor, w2ds, valueBinIndices, parallelLinearLayer,
                  bestPars, bestTestLoss, testLosses],
                 imageDir + targetCleaned + timeString + '.pickle')

# Overwrite existing pars with best tune values
for pi, p in enumerate(bestPars):
    pars[pi].data = p
if hawkesrd is not None:
    jh.resetHawkesTensors(hawkesrd, hawkespd, waveArrayTensor)


# Apply to hold out set: overwrite the test set with hold out
holdoutObjects = jrd.load_dat4_dataset(dataSetDict=details, iterString='holdout', tvShape=np.array([None, Tmax, None]),eventBins=Ebins)
_, _, testLbub, testParams, testPtsTimesValuesTensor, _ = holdoutObjects[0]
Ntest = testParams[0]
testWatx, testCountsInWatxTensor, testDistsInWatxTensor = f2m.derivedTensors(testLbub, target, testPtsTimesValuesTensor, targetSteps)
_, _, _, testValueBinIndices, _ =     f2m.getImageTensors(target, events, wsizes, testPtsTimesValuesTensor, scaleSteps)
testSsmaps = f2m.getSSMaps(testPtsTimesValuesTensor, waveArrayTensor, testWatx)
print('Holdout performance: ')
testLoss, testHitsArray, testAreaArray, testLcLayerArray, testRlLayerArray, testTtimesArray =     runStep(t, Ntest, Tmax, targetSteps, None, events, testPtsTimesValuesTensor,
            waveArrayTensor, hd, testSsmaps, testCountsInWatxTensor, testDistsInWatxTensor,
            w2ds, testValueBinIndices,
            # indexBatch=np.random.choice(Ntest, testBatch, False),
            coefficientPenalties=coefficientPenalties,
            hadamardCensor=hadamardCensor, overrideReconstructionDict=hawkesrd, overrideParametersDict=hawkespd,
            verbose=True)

# Bootstrap the tune (and held out set when decide to access) set to get confidence intervals
# this is a fixed model bootstrap, i.e. once learned \hat{f} is fixed, you want to know confidence in log likelihood on holdout test
# this does not assess performance of the learning procedure; it only characterizes the performance of the resulting model \hat{f}
bsn = 1000
blogli = np.full(bsn,np.nan)
for i in np.arange(bsn):
    bn = testHitsArray.squeeze().size()[0]
    sample = np.random.choice(bn,bn)
    blogli[i] = (testHitsArray.squeeze().data.numpy()[sample] - testAreaArray.squeeze().data.numpy()[sample]).sum()/bn
blb, bub = np.percentile(blogli,[2.5, 97.5])
print('[2.5%,97.5%]: ', blb, bub)


testBaseRate = (~np.isnan(testTtimesArray.data.numpy())).sum() / (testLbub[:,1]-testLbub[:,0]).sum()
print('Average rate (holdout): ' + str(testBaseRate))
print('Average NLL guessing (holdout): ' +       str(testBaseRate*(testLbub[:,1]-testLbub[:,0]).sum()/testLbub.shape[0] - np.log(testBaseRate)*testBaseRate*(testLbub[:,1]-testLbub[:,0]).sum()/testLbub.shape[0]))


# See coefficents as a function of censorStep
_, _, _, testValueBinIndices, _ =     f2m.getImageTensors(target, events, wsizes, testPtsTimesValuesTensor, scaleSteps)
testLoss, testHitsArray, testAreaArray, testLcLayerArray, testRlLayerArray, testTtimesArray = runStep(
    t, Ntest, Tmax, targetSteps, None, events, testPtsTimesValuesTensor, waveArrayTensor, hd, testSsmaps, testCountsInWatxTensor, testDistsInWatxTensor, w2ds, testValueBinIndices, coefficientPenalties=coefficientPenalties, hadamardCensor=hadamardCensor, overrideReconstructionDict=hawkesrd, overrideParametersDict=hawkespd, verbose=True)  # use waveArrayTensor no testWaveArrayTensor because waveArrayTensor holds train set modified wavelet reconstruction parameters
print('Tune performance as a function of censor step')
print((testHitsArray - testAreaArray).sum(0).neg_()/Ntest)


# random helpers
torch.cat([testPtsTimesValuesTensor[e][3,:,0].unsqueeze(1) for e in testPtsTimesValuesTensor.keys()],1)
testLcLayerArray[np.argsort(batchis),:].squeeze(1)

###  Loading from a saved file
# 1. load from save file
# vals = f2m.loadJcwModel('replace')
# t, N, Tmax, Ebins, targetSteps, optimizer, events, ptsTimesValuesTensor, waveArrayTensor, hd, ssmaps, countsInWatxTensor, distsInWatxTensor, w2ds, valueBinIndices, parallelLinearLayer, bestPars, bestTestLoss, testLosses = vals
# 2. ensure the wsizes gets overwrote to match this data

# Test set {log likelihood, FP per TP} as a function of forecast window

