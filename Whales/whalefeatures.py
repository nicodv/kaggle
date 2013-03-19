#!/usr/bin/python

import os
import aifc
import numpy as np
import yaafelib as yl
import re
import pandas as pd

traindir ='/home/nico/Code/datasets/Kaggle/Whales/train'
testdir  ='/home/nico/Code/datasets/Kaggle/Whales/test'
datdir   ='/home/nico/Code/datasets/Kaggle/Whales'

SAMPLE_RATE = 2000
SAMPLE_LENGTH = 2

def read_samples(dir):
    allSigs = np.zeros( (len(os.listdir(dir)),SAMPLE_LENGTH*SAMPLE_RATE) )
    filenumbers = []
    for cnt, filename in enumerate(os.listdir(dir)):
        if os.path.isfile(os.path.join(dir, filename)):
            filenumbers.append(int(re.findall('[0-9]+',filename)[0]))
            sample = aifc.open(os.path.join(dir, filename), 'r')
            nframes = sample.getnframes()
            strSig = sample.readframes(nframes)
            allSigs[cnt,:] = np.fromstring(strSig, np.short).byteswap()
            sample.close()
    return filenumbers, allSigs

def read_targets():
    targets = pd.read_csv(os.path.join(datdir,'train.csv'))
    targets.clip_name = targets.clip_name.map(lambda x: x.replace('.aiff',''))
    targets.clip_name = targets.clip_name.map(lambda x: int(x.replace('train','')))
    targets = targets.set_index('clip_name')
    targets = targets.sort()
    
    return targets.label

def extract_audio_features(sigdata):
    window = 'Hanning'
    block = 120
    step = 60
    
    fp = yl.FeaturePlan(sample_rate=SAMPLE_RATE)
    fp.addFeature('CDOD: ComplexDomainOnsetDetection FFTWindow=%s blockSize=%d stepSize=%d' % (window, block, step))
    fp.addFeature('LPC: LPC LPCNbCoeffs=4 blockSize=%d stepSize=%d' % (block, step))
    fp.addFeature('MagnSpec: MagnitudeSpectrum FFTWindow=%s blockSize=%d stepSize=%d' % (window, block, step))
    fp.addFeature('MelSpec: MelSpectrum FFTWindow=%s MelMaxFreq=700. MelMinFreq=30. MelNbFilters=40 blockSize=%d stepSize=%d' % (window, block, step))
    fp.addFeature('MFCC: MFCC CepsIgnoreFirstCoeff=1 CepsNbCoeffs=12 FFTWindow=%s MelMaxFreq=700. MelMinFreq=30. MelNbFilters=40 blockSize=%d stepSize=%d' % (window, block, step))
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
        
        # trim some uninteresting frequencies off spectrum
        feats[cnt]['MagnSpec'] = feats[cnt]['MagnSpec'][:,2:36]
        
    return feats

if __name__ == '__main__':
    for curstr in ('train','test'):
        # read samples and store file numbers
        numbers, sigs = read_samples(eval(curstr+'dir'))
        
        if curstr == 'train':
            # add DCLDE 2013 Workshop Dataset data
            numbers = [x+6671 for x in numbers]
            extnumbers = range(1,6672)
            numbers.extend(extnumbers)
            extradata = np.genfromtxt(datdir+'/extra/signals.csv', delimiter=',')
            sigs = np.concatenate((sigs,extradata))
            assert len(numbers)==len(sigs)
        
        # make sure the data is sorted according to file number
        numbers = np.array(numbers)
        sigs = sigs[numbers.argsort()]
        
        feats = extract_audio_features(sigs)
        
        # split into 3 data sets
        spectrum = np.array([x['MagnSpec'] for x in feats])
        melspectrum = np.array([x['MelSpec'] for x in feats])
        specfeat = np.array([np.concatenate((x['MFCC'],x['CDOD'],x['LPC'],x['SF'], \
                    x['SpecStats'],x['SpecSlope'],x['SpecVar']),axis=1) for x in feats])
        
        np.save(os.path.join(datdir,curstr+'spectrum'), spectrum)
        np.save(os.path.join(datdir,curstr+'melspectrum'), melspectrum)
        np.save(os.path.join(datdir,curstr+'specfeat'), specfeat)
    
    targets = read_targets()
    # add DCLDE 2013 Workshop Dataset labels
    extlabels = pd.Series(np.ones(6671))
    targets = pd.concat([extlabels,targets])
    # convert to one-hot numpy array
    targets = np.array((targets,-targets+1)).T
    np.save(os.path.join(datdir,'targets'), targets)
