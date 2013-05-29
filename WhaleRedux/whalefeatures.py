#!/usr/bin/python

import os
import aifc
import numpy as np
import yaafelib as yl
import re
import pandas as pd

traindir ='/home/nico/datasets/Kaggle/WhaleRedux/train2'
testdir  ='/home/nico/datasets/Kaggle/WhaleRedux/test2'
datdir   ='/home/nico/datasets/Kaggle/WhaleRedux'

SAMPLE_RATE = 2000
SAMPLE_LENGTH = 2

def read_samples(dir):
    allSigs = np.zeros( (len(os.listdir(dir)),SAMPLE_LENGTH*SAMPLE_RATE) )
    filenames = []
    for cnt, filename in enumerate(os.listdir(dir)):
        if os.path.isfile(os.path.join(dir, filename)):
            filenames.append(filename)
            sample = aifc.open(os.path.join(dir, filename), 'r')
            nframes = sample.getnframes()
            strSig = sample.readframes(nframes)
            sig = np.fromstring(strSig, np.short).byteswap()
            if nframes > 4000:
                sig = sig[nframes-4000//2:(nframes-4000//2)+4000]
            elif nframes < 4000:
                sig = np.append(sig,np.zeros(4000-nframes))
            allSigs[cnt,:] = sig
            sample.close()
    return filenames, allSigs

def read_targets():
    targets = []
    for cnt, filename in enumerate(os.listdir(traindir)):
        if os.path.isfile(os.path.join(traindir, filename)):
            targets.append(int(re.search('(ms_TRAIN[0-9]*\_([0-9]*))',filename).group(2)))
    return targets

def extract_audio_features(sigdata):
    '''Extracts a bunch of audio features using YAAFE
    '''
    window = 'Hanning'
    block = 120
    step = 60
    
    fp = yl.FeaturePlan(sample_rate=SAMPLE_RATE)
    fp.addFeature('CDOD: ComplexDomainOnsetDetection FFTWindow=%s blockSize=%d stepSize=%d' % (window, block, step))
    fp.addFeature('LPC: LPC LPCNbCoeffs=4 blockSize=%d stepSize=%d' % (block, step))
    fp.addFeature('MelSpec: MelSpectrum FFTWindow=%s MelMaxFreq=600 MelMinFreq=30 MelNbFilters=40 blockSize=%d stepSize=%d' % (window, block, step))
    fp.addFeature('MFCC: MFCC CepsIgnoreFirstCoeff=1 CepsNbCoeffs=12 FFTWindow=%s MelMaxFreq=600 MelMinFreq=30 MelNbFilters=40 blockSize=%d stepSize=%d' % (window, block, step))
    fp.addFeature('SF: SpectralFlux FFTWindow=%s FluxSupport=Increase blockSize=%d stepSize=%d' % (window, block, step))
    fp.addFeature('SpecStats: SpectralShapeStatistics FFTWindow=%s blockSize=%d stepSize=%d' % (window, block, step))
    fp.addFeature('SpecSlope: SpectralSlope FFTWindow=%s blockSize=%d stepSize=%d' % (window, block, step))
    fp.addFeature('SpecVar: SpectralVariation FFTWindow=%s blockSize=%d stepSize=%d' % (window, block, step))
    df = fp.getDataFlow()
    # df.display()
    
    engine = yl.Engine()
    engine.load(df)
    
    feats = []
    for cnt in range(sigdata.shape[0]):
        signal = np.reshape(sigdata[cnt,:],[1,-1])
        feats.append(engine.processAudio(signal))
    
    return feats

def _remove_bias(data, window=50):
    '''Remove bias from signals
        (in some data there are low-frequency
        waves that we get rid of using a moving average)
    '''
    df = pd.DataFrame(data)
    df = df - pd.rolling_mean(df, window, axis=1, min_periods=0)
    return np.array(df)

if __name__ == '__main__':
    
    for curstr in ('train','test'):
        # read samples and store file numbers
        names, sigs = read_samples(eval(curstr+'dir'))
        
        # original data is pretty clean, but still remove mean
        sigs = sigs - np.mean(sigs, axis=1, keepdims=True)
        
        # make sure the data is sorted according to file number
        # (necessary, since sorting is alphanumeric: 1, 10, 100, 2, etc.)
        #names = np.array(names)
        #sigs = sigs[names.argsort()]
        
        # standardize all signals
        sigs = sigs / np.std(sigs, axis=1, keepdims=True)
        
        # now we can extract features
        feats = extract_audio_features(sigs)
        
        # split into 2 data sets with 2D and 1D data, respectively
        melspectrum = np.array([x['MelSpec'] for x in feats])
        specfeat = np.array([np.concatenate((x['MFCC'],x['CDOD'],x['LPC'],x['SF'], \
                    x['SpecStats'],x['SpecSlope'],x['SpecVar']),axis=1) for x in feats])
        
        np.save(os.path.join(datdir,curstr+'melspectrum'), melspectrum)
        np.save(os.path.join(datdir,curstr+'specfeat'), specfeat)
    
    targets = read_targets()
    targets = np.array(targets)
    
    # convert to one-hot numpy array
    targets = np.array((targets,-targets+1)).T
    
    np.save(os.path.join(datdir,'targets'), targets)
