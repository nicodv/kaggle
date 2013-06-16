#!/usr/bin/python

import os
import aifc
import numpy as np
import yaafelib as yl
import re

datdir   = '/home/nico/datasets/Kaggle/WhaleRedux'
traindir = os.path.join(datdir,'train2')
testdir  = os.path.join(datdir,'test2')

EXTRA_DATA = False

SAMPLE_RATE = 2000
SAMPLE_LENGTH = 2

def read_samples(dir):
    allSigs = np.zeros( (len(os.listdir(dir)),SAMPLE_LENGTH*SAMPLE_RATE) )
    filenames = []
    targets = []
    for cnt, filename in enumerate(os.listdir(dir)):
        if os.path.isfile(os.path.join(dir, filename)):
            filenames.append(filename)
            if dir == traindir:
                targets.append(int(re.search('(ms_TRAIN[0-9]*\_([0-9]*))',filename).group(2)))
            else:
                targets.append(0)
            sample = aifc.open(os.path.join(dir, filename), 'r')
            nframes = sample.getnframes()
            strSig = sample.readframes(nframes)
            sig = np.fromstring(strSig, np.short).byteswap()
            if nframes > 4000:
                sig = sig[nframes-4000//2:(nframes-4000//2)+4000]
            elif nframes < 4000:
                # appending is OK instead of symmetrical prepending and appending,
                # since we're going to sample patches anyway
                sig = np.append(sig,np.zeros(4000-nframes))
            allSigs[cnt,:] = sig
            sample.close()
    
    targets = np.array(targets)
    
    return filenames, targets, allSigs

def extract_audio_features(sigdata):
    '''Extracts a bunch of audio features using YAAFE
    '''
    window = 'Hanning'
    # using 80 / 40 here produces NaNs in mel spectrum, for some reason
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

if __name__ == '__main__':
    
    for curstr in ('train','test'):
        # read samples and store file numbers
        names, targs, sigs = read_samples(eval(curstr+'dir'))
        
        #sort everything properly
        names = np.array(names)
        if curstr == 'train':
            targets = targs[names.argsort()]
        sigs = sigs[names.argsort()]
        names.sort()
        
        # save names for submissions
        if curstr == 'test':
            np.save(os.path.join(datdir,'filenames'), names)
        
        # standardize all signals
        sigs = sigs - np.mean(sigs, axis=1, keepdims=True)
        sigs = sigs / np.std(sigs, axis=1, keepdims=True)
        
        # now we can extract features
        feats = extract_audio_features(sigs)
        
        # split into 2 data sets with 2D and 1D data, respectively
        melspectrum = np.array([x['MelSpec'] for x in feats])
        specfeat = np.array([np.concatenate((x['MFCC'],x['CDOD'],x['LPC'],x['SF'], \
                    x['SpecStats'],x['SpecSlope'],x['SpecVar']),axis=1) for x in feats])
        
        if EXTRA_DATA:
            if curstr == 'train':
                # generate extra training data by adding no-whale data to whale data
                for ii in range(10000):
                    if np.remainder(ii,10) == 0:
                        print(str(ii))
                    # pick random examples of whale and no-whale examples
                    indt = np.random.choice(np.where(targets==1)[0], 1)[0]
                    indf = np.random.choice(np.where(targets==0)[0], 1)[0]
                    # construct new training example
                    xexmel = melspectrum[indt] + (np.random.rand()/2)*melspectrum[indf]
                    xexsf = specfeat[indt] + (np.random.rand()/2)*specfeat[indf]
                    # inserting to keep examples ordered (note: now overestimating autocorrelation)
                    melspectrum = np.insert(melspectrum, indt, xexmel, axis=0)
                    specfeat = np.insert(specfeat, indt, xexsf, axis=0)
                    targets = np.insert(targets, indt, 1, axis=0)
        
        np.save(os.path.join(datdir,curstr+'_melspectrum'), melspectrum)
        np.save(os.path.join(datdir,curstr+'_specfeat'), specfeat)
    
    # convert to one-hot numpy array
    targets = np.array((targets,-targets+1)).T
    
    np.save(os.path.join(datdir,'targets'), targets)
    
