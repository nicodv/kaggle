#!/usr/bin/python2

import os
from pylearn2.datasets import preprocessing
from pylearn2.utils import serial
from collections import OrderedDict
import numpy as np
from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix, DefaultViewConverter
from PIL import Image

import GalaxyZoo.gzdeepdata

DATA_DIR = '/home/nico/Data/GalaxyZoo/'

SUBMODEL = 2


class GZData(DenseDesignMatrix):

    def __init__(self, which_set, start=None, stop=None, axes=('c', 0, 1, 'b')):
        assert which_set in ['training', 'test']

        filenames = np.load(os.path.join(DATA_DIR, 'filenumbers.npy'))
        if which_set == 'training':
            filenames = filenames[0]
        elif which_set == 'test':
            filenames = filenames[1]

        self.patch_size = (128, 128)

        # downsample data?
        self.scale_factor = 2

        if start is not None:
            assert start >= 0
            assert start < stop
            assert stop <= 61577
            filenames = filenames[start:stop + 1]

        trainx = np.zeros((2 * len(filenames), int(self.patch_size[0] / self.scale_factor),
                           int(self.patch_size[1] / self.scale_factor), 3))
        for ii, fn in enumerate(filenames):
            for jj, itype in enumerate(('raw', 'proc')):
                im = Image.open(DATA_DIR + 'images_' + which_set + '_proc/' + str(fn) + '_' + itype + '.png')

                # resize
                if self.scale_factor not in (None, 1, 1.):
                    # why are these floats?
                    w, h = [int(x) for x in im.size]
                    im = im.resize((int(w / self.scale_factor), int(h / self.scale_factor)), Image.ANTIALIAS)

                # from here on, work with 2D numpy array
                im = np.squeeze(np.array(im, dtype=np.uint8))

                trainx[2 * ii + jj, :, :, :] = im

        trainx = np.reshape(trainx, (trainx.shape[0], np.prod(trainx.shape[1:])))
        trainx = trainx.astype('float32')

        if SUBMODEL == 2:
            if which_set == 'test':
                y = np.zeros((trainx.shape[0], 8)).astype('float32')
            else:
                y = np.load(DATA_DIR+'targets.npy').astype('float32')
                if start is None:
                    start = 0
                    stop = trainx.shape[0] - 1
                # 14:15 is zodat de array 1 dimensie houdt in die richting
                y = np.hstack((y[start:stop + 1, 14:15], y[start:stop + 1, 18:25]))
                y = np.repeat(y, 2, axis=0)
        elif SUBMODEL == 1:
            if which_set == 'test':
                y = np.zeros((trainx.shape[0], 159)).astype('float32')
            else:
                y = np.load(DATA_DIR+'targets.npy').astype('float32')
                if start is None:
                    start = 0
                    stop = trainx.shape[0] - 1
                y = np.hstack((
                    # 1.1 + 7.1/.2/.3
                    y[start:stop + 1, 15:18],
                    # 1.2 + 2.1 + 9.1/.2/.3
                    y[start:stop + 1, 25:28],
                    # 1.2 + 2.2 + 3.1 + 4.1 + 10.1 + 11.1 + 5.1/.2/.3/.4
                    y[start:stop + 1, 5:6] * (y[start:stop + 1, 7:8] / np.sum(y[start:stop + 1, 7:9], axis=1, keepdims=True)) * (y[start:stop + 1, 28:29] / np.sum(y[start:stop + 1, 28:31], axis=1, keepdims=True)) * (y[start:stop + 1, 31:32] / np.sum(y[start:stop + 1, 31:37], axis=1, keepdims=True)) * (y[start:stop + 1, 9:13] / np.sum(y[start:stop + 1, 9:13], axis=1, keepdims=True)),
                    # 1.2 + 2.2 + 3.1 + 4.1 + 10.1 + 11.2 + 5.1/.2/.3/.4
                    y[start:stop + 1, 5:6] * (y[start:stop + 1, 7:8] / np.sum(y[start:stop + 1, 7:9], axis=1, keepdims=True)) * (y[start:stop + 1, 28:29] / np.sum(y[start:stop + 1, 28:31], axis=1, keepdims=True)) * (y[start:stop + 1, 32:33] / np.sum(y[start:stop + 1, 31:37], axis=1, keepdims=True)) * (y[start:stop + 1, 9:13] / np.sum(y[start:stop + 1, 9:13], axis=1, keepdims=True)),
                    # 1.2 + 2.2 + 3.1 + 4.1 + 10.1 + 11.3 + 5.1/.2/.3/.4
                    y[start:stop + 1, 5:6] * (y[start:stop + 1, 7:8] / np.sum(y[start:stop + 1, 7:9], axis=1, keepdims=True)) * (y[start:stop + 1, 28:29] / np.sum(y[start:stop + 1, 28:31], axis=1, keepdims=True)) * (y[start:stop + 1, 33:34] / np.sum(y[start:stop + 1, 31:37], axis=1, keepdims=True)) * (y[start:stop + 1, 9:13] / np.sum(y[start:stop + 1, 9:13], axis=1, keepdims=True)),
                    # 1.2 + 2.2 + 3.1 + 4.1 + 10.1 + 11.4 + 5.1/.2/.3/.4
                    y[start:stop + 1, 5:6] * (y[start:stop + 1, 7:8] / np.sum(y[start:stop + 1, 7:9], axis=1, keepdims=True)) * (y[start:stop + 1, 28:29] / np.sum(y[start:stop + 1, 28:31], axis=1, keepdims=True)) * (y[start:stop + 1, 34:35] / np.sum(y[start:stop + 1, 31:37], axis=1, keepdims=True)) * (y[start:stop + 1, 9:13] / np.sum(y[start:stop + 1, 9:13], axis=1, keepdims=True)),
                    # 1.2 + 2.2 + 3.1 + 4.1 + 10.1 + 11.5 + 5.1/.2/.3/.4
                    y[start:stop + 1, 5:6] * (y[start:stop + 1, 7:8] / np.sum(y[start:stop + 1, 7:9], axis=1, keepdims=True)) * (y[start:stop + 1, 28:29] / np.sum(y[start:stop + 1, 28:31], axis=1, keepdims=True)) * (y[start:stop + 1, 35:36] / np.sum(y[start:stop + 1, 31:37], axis=1, keepdims=True)) * (y[start:stop + 1, 9:13] / np.sum(y[start:stop + 1, 9:13], axis=1, keepdims=True)),
                    # 1.2 + 2.2 + 3.1 + 4.1 + 10.1 + 11.6 + 5.1/.2/.3/.4
                    y[start:stop + 1, 5:6] * (y[start:stop + 1, 7:8] / np.sum(y[start:stop + 1, 7:9], axis=1, keepdims=True)) * (y[start:stop + 1, 28:29] / np.sum(y[start:stop + 1, 28:31], axis=1, keepdims=True)) * (y[start:stop + 1, 36:37] / np.sum(y[start:stop + 1, 31:37], axis=1, keepdims=True)) * (y[start:stop + 1, 9:13] / np.sum(y[start:stop + 1, 9:13], axis=1, keepdims=True)),
                    # 1.2 + 2.2 + 3.1 + 4.1 + 10.2 + 11.1 + 5.1/.2/.3/.4
                    y[start:stop + 1, 5:6] * (y[start:stop + 1, 7:8] / np.sum(y[start:stop + 1, 7:9], axis=1, keepdims=True)) * (y[start:stop + 1, 29:30] / np.sum(y[start:stop + 1, 28:31], axis=1, keepdims=True)) * (y[start:stop + 1, 31:32] / np.sum(y[start:stop + 1, 31:37], axis=1, keepdims=True)) * (y[start:stop + 1, 9:13] / np.sum(y[start:stop + 1, 9:13], axis=1, keepdims=True)),
                    # 1.2 + 2.2 + 3.1 + 4.1 + 10.2 + 11.2 + 5.1/.2/.3/.4
                    y[start:stop + 1, 5:6] * (y[start:stop + 1, 7:8] / np.sum(y[start:stop + 1, 7:9], axis=1, keepdims=True)) * (y[start:stop + 1, 29:30] / np.sum(y[start:stop + 1, 28:31], axis=1, keepdims=True)) * (y[start:stop + 1, 32:33] / np.sum(y[start:stop + 1, 31:37], axis=1, keepdims=True)) * (y[start:stop + 1, 9:13] / np.sum(y[start:stop + 1, 9:13], axis=1, keepdims=True)),
                    # 1.2 + 2.2 + 3.1 + 4.1 + 10.2 + 11.3 + 5.1/.2/.3/.4
                    y[start:stop + 1, 5:6] * (y[start:stop + 1, 7:8] / np.sum(y[start:stop + 1, 7:9], axis=1, keepdims=True)) * (y[start:stop + 1, 29:30] / np.sum(y[start:stop + 1, 28:31], axis=1, keepdims=True)) * (y[start:stop + 1, 33:34] / np.sum(y[start:stop + 1, 31:37], axis=1, keepdims=True)) * (y[start:stop + 1, 9:13] / np.sum(y[start:stop + 1, 9:13], axis=1, keepdims=True)),
                    # 1.2 + 2.2 + 3.1 + 4.1 + 10.2 + 11.4 + 5.1/.2/.3/.4
                    y[start:stop + 1, 5:6] * (y[start:stop + 1, 7:8] / np.sum(y[start:stop + 1, 7:9], axis=1, keepdims=True)) * (y[start:stop + 1, 29:30] / np.sum(y[start:stop + 1, 28:31], axis=1, keepdims=True)) * (y[start:stop + 1, 34:35] / np.sum(y[start:stop + 1, 31:37], axis=1, keepdims=True)) * (y[start:stop + 1, 9:13] / np.sum(y[start:stop + 1, 9:13], axis=1, keepdims=True)),
                    # 1.2 + 2.2 + 3.1 + 4.1 + 10.2 + 11.5 + 5.1/.2/.3/.4
                    y[start:stop + 1, 5:6] * (y[start:stop + 1, 7:8] / np.sum(y[start:stop + 1, 7:9], axis=1, keepdims=True)) * (y[start:stop + 1, 29:30] / np.sum(y[start:stop + 1, 28:31], axis=1, keepdims=True)) * (y[start:stop + 1, 35:36] / np.sum(y[start:stop + 1, 31:37], axis=1, keepdims=True)) * (y[start:stop + 1, 9:13] / np.sum(y[start:stop + 1, 9:13], axis=1, keepdims=True)),
                    # 1.2 + 2.2 + 3.1 + 4.1 + 10.2 + 11.6 + 5.1/.2/.3/.4
                    y[start:stop + 1, 5:6] * (y[start:stop + 1, 7:8] / np.sum(y[start:stop + 1, 7:9], axis=1, keepdims=True)) * (y[start:stop + 1, 29:30] / np.sum(y[start:stop + 1, 28:31], axis=1, keepdims=True)) * (y[start:stop + 1, 36:37] / np.sum(y[start:stop + 1, 31:37], axis=1, keepdims=True)) * (y[start:stop + 1, 9:13] / np.sum(y[start:stop + 1, 9:13], axis=1, keepdims=True)),
                    # 1.2 + 2.2 + 3.1 + 4.1 + 10.3 + 11.1 + 5.1/.2/.3/.4
                    y[start:stop + 1, 5:6] * (y[start:stop + 1, 7:8] / np.sum(y[start:stop + 1, 7:9], axis=1, keepdims=True)) * (y[start:stop + 1, 30:31] / np.sum(y[start:stop + 1, 28:31], axis=1, keepdims=True)) * (y[start:stop + 1, 31:32] / np.sum(y[start:stop + 1, 31:37], axis=1, keepdims=True)) * (y[start:stop + 1, 9:13] / np.sum(y[start:stop + 1, 9:13], axis=1, keepdims=True)),
                    # 1.2 + 2.2 + 3.1 + 4.1 + 10.3 + 11.2 + 5.1/.2/.3/.4
                    y[start:stop + 1, 5:6] * (y[start:stop + 1, 7:8] / np.sum(y[start:stop + 1, 7:9], axis=1, keepdims=True)) * (y[start:stop + 1, 30:31] / np.sum(y[start:stop + 1, 28:31], axis=1, keepdims=True)) * (y[start:stop + 1, 32:33] / np.sum(y[start:stop + 1, 31:37], axis=1, keepdims=True)) * (y[start:stop + 1, 9:13] / np.sum(y[start:stop + 1, 9:13], axis=1, keepdims=True)),
                    # 1.2 + 2.2 + 3.1 + 4.1 + 10.3 + 11.3 + 5.1/.2/.3/.4
                    y[start:stop + 1, 5:6] * (y[start:stop + 1, 7:8] / np.sum(y[start:stop + 1, 7:9], axis=1, keepdims=True)) * (y[start:stop + 1, 30:31] / np.sum(y[start:stop + 1, 28:31], axis=1, keepdims=True)) * (y[start:stop + 1, 33:34] / np.sum(y[start:stop + 1, 31:37], axis=1, keepdims=True)) * (y[start:stop + 1, 9:13] / np.sum(y[start:stop + 1, 9:13], axis=1, keepdims=True)),
                    # 1.2 + 2.2 + 3.1 + 4.1 + 10.3 + 11.4 + 5.1/.2/.3/.4
                    y[start:stop + 1, 5:6] * (y[start:stop + 1, 7:8] / np.sum(y[start:stop + 1, 7:9], axis=1, keepdims=True)) * (y[start:stop + 1, 30:31] / np.sum(y[start:stop + 1, 28:31], axis=1, keepdims=True)) * (y[start:stop + 1, 34:35] / np.sum(y[start:stop + 1, 31:37], axis=1, keepdims=True)) * (y[start:stop + 1, 9:13] / np.sum(y[start:stop + 1, 9:13], axis=1, keepdims=True)),
                    # 1.2 + 2.2 + 3.1 + 4.1 + 10.3 + 11.5 + 5.1/.2/.3/.4
                    y[start:stop + 1, 5:6] * (y[start:stop + 1, 7:8] / np.sum(y[start:stop + 1, 7:9], axis=1, keepdims=True)) * (y[start:stop + 1, 30:31] / np.sum(y[start:stop + 1, 28:31], axis=1, keepdims=True)) * (y[start:stop + 1, 35:36] / np.sum(y[start:stop + 1, 31:37], axis=1, keepdims=True)) * (y[start:stop + 1, 9:13] / np.sum(y[start:stop + 1, 9:13], axis=1, keepdims=True)),
                    # 1.2 + 2.2 + 3.1 + 4.1 + 10.3 + 11.6 + 5.1/.2/.3/.4
                    y[start:stop + 1, 5:6] * (y[start:stop + 1, 7:8] / np.sum(y[start:stop + 1, 7:9], axis=1, keepdims=True)) * (y[start:stop + 1, 30:31] / np.sum(y[start:stop + 1, 28:31], axis=1, keepdims=True)) * (y[start:stop + 1, 36:37] / np.sum(y[start:stop + 1, 31:37], axis=1, keepdims=True)) * (y[start:stop + 1, 9:13] / np.sum(y[start:stop + 1, 9:13], axis=1, keepdims=True)),
                    # 1.2 + 2.2 + 3.1 + 4.2 + 5.1/.2/.3/.4
                    y[start:stop + 1, 5:6] * (y[start:stop + 1, 8:9] / np.sum(y[start:stop + 1, 7:9], axis=1, keepdims=True)) * (y[start:stop + 1, 9:13] / np.sum(y[start:stop + 1, 9:13], axis=1, keepdims=True)),
                    # 1.2 + 2.2 + 3.2 + 4.1 + 10.1 + 11.1 + 5.1/.2/.3/.4
                    y[start:stop + 1, 6:7] * (y[start:stop + 1, 7:8] / np.sum(y[start:stop + 1, 7:9], axis=1, keepdims=True)) * (y[start:stop + 1, 28:29] / np.sum(y[start:stop + 1, 28:31], axis=1, keepdims=True)) * (y[start:stop + 1, 31:32] / np.sum(y[start:stop + 1, 31:37], axis=1, keepdims=True)) * (y[start:stop + 1, 9:13] / np.sum(y[start:stop + 1, 9:13], axis=1, keepdims=True)),
                    # 1.2 + 2.2 + 3.2 + 4.1 + 10.1 + 11.2 + 5.1/.2/.3/.4
                    y[start:stop + 1, 6:7] * (y[start:stop + 1, 7:8] / np.sum(y[start:stop + 1, 7:9], axis=1, keepdims=True)) * (y[start:stop + 1, 28:29] / np.sum(y[start:stop + 1, 28:31], axis=1, keepdims=True)) * (y[start:stop + 1, 32:33] / np.sum(y[start:stop + 1, 31:37], axis=1, keepdims=True)) * (y[start:stop + 1, 9:13] / np.sum(y[start:stop + 1, 9:13], axis=1, keepdims=True)),
                    # 1.2 + 2.2 + 3.2 + 4.1 + 10.1 + 11.3 + 5.1/.2/.3/.4
                    y[start:stop + 1, 6:7] * (y[start:stop + 1, 7:8] / np.sum(y[start:stop + 1, 7:9], axis=1, keepdims=True)) * (y[start:stop + 1, 28:29] / np.sum(y[start:stop + 1, 28:31], axis=1, keepdims=True)) * (y[start:stop + 1, 33:34] / np.sum(y[start:stop + 1, 31:37], axis=1, keepdims=True)) * (y[start:stop + 1, 9:13] / np.sum(y[start:stop + 1, 9:13], axis=1, keepdims=True)),
                    # 1.2 + 2.2 + 3.2 + 4.1 + 10.1 + 11.4 + 5.1/.2/.3/.4
                    y[start:stop + 1, 6:7] * (y[start:stop + 1, 7:8] / np.sum(y[start:stop + 1, 7:9], axis=1, keepdims=True)) * (y[start:stop + 1, 28:29] / np.sum(y[start:stop + 1, 28:31], axis=1, keepdims=True)) * (y[start:stop + 1, 34:35] / np.sum(y[start:stop + 1, 31:37], axis=1, keepdims=True)) * (y[start:stop + 1, 9:13] / np.sum(y[start:stop + 1, 9:13], axis=1, keepdims=True)),
                    # 1.2 + 2.2 + 3.2 + 4.1 + 10.1 + 11.5 + 5.1/.2/.3/.4
                    y[start:stop + 1, 6:7] * (y[start:stop + 1, 7:8] / np.sum(y[start:stop + 1, 7:9], axis=1, keepdims=True)) * (y[start:stop + 1, 28:29] / np.sum(y[start:stop + 1, 28:31], axis=1, keepdims=True)) * (y[start:stop + 1, 35:36] / np.sum(y[start:stop + 1, 31:37], axis=1, keepdims=True)) * (y[start:stop + 1, 9:13] / np.sum(y[start:stop + 1, 9:13], axis=1, keepdims=True)),
                    # 1.2 + 2.2 + 3.2 + 4.1 + 10.1 + 11.6 + 5.1/.2/.3/.4
                    y[start:stop + 1, 6:7] * (y[start:stop + 1, 7:8] / np.sum(y[start:stop + 1, 7:9], axis=1, keepdims=True)) * (y[start:stop + 1, 28:29] / np.sum(y[start:stop + 1, 28:31], axis=1, keepdims=True)) * (y[start:stop + 1, 36:37] / np.sum(y[start:stop + 1, 31:37], axis=1, keepdims=True)) * (y[start:stop + 1, 9:13] / np.sum(y[start:stop + 1, 9:13], axis=1, keepdims=True)),
                    # 1.2 + 2.2 + 3.2 + 4.1 + 10.2 + 11.1 + 5.1/.2/.3/.4
                    y[start:stop + 1, 6:7] * (y[start:stop + 1, 7:8] / np.sum(y[start:stop + 1, 7:9], axis=1, keepdims=True)) * (y[start:stop + 1, 29:30] / np.sum(y[start:stop + 1, 28:31], axis=1, keepdims=True)) * (y[start:stop + 1, 31:32] / np.sum(y[start:stop + 1, 31:37], axis=1, keepdims=True)) * (y[start:stop + 1, 9:13] / np.sum(y[start:stop + 1, 9:13], axis=1, keepdims=True)),
                    # 1.2 + 2.2 + 3.2 + 4.1 + 10.2 + 11.2 + 5.1/.2/.3/.4
                    y[start:stop + 1, 6:7] * (y[start:stop + 1, 7:8] / np.sum(y[start:stop + 1, 7:9], axis=1, keepdims=True)) * (y[start:stop + 1, 29:30] / np.sum(y[start:stop + 1, 28:31], axis=1, keepdims=True)) * (y[start:stop + 1, 32:33] / np.sum(y[start:stop + 1, 31:37], axis=1, keepdims=True)) * (y[start:stop + 1, 9:13] / np.sum(y[start:stop + 1, 9:13], axis=1, keepdims=True)),
                    # 1.2 + 2.2 + 3.2 + 4.1 + 10.2 + 11.3 + 5.1/.2/.3/.4
                    y[start:stop + 1, 6:7] * (y[start:stop + 1, 7:8] / np.sum(y[start:stop + 1, 7:9], axis=1, keepdims=True)) * (y[start:stop + 1, 29:30] / np.sum(y[start:stop + 1, 28:31], axis=1, keepdims=True)) * (y[start:stop + 1, 33:34] / np.sum(y[start:stop + 1, 31:37], axis=1, keepdims=True)) * (y[start:stop + 1, 9:13] / np.sum(y[start:stop + 1, 9:13], axis=1, keepdims=True)),
                    # 1.2 + 2.2 + 3.2 + 4.1 + 10.2 + 11.4 + 5.1/.2/.3/.4
                    y[start:stop + 1, 6:7] * (y[start:stop + 1, 7:8] / np.sum(y[start:stop + 1, 7:9], axis=1, keepdims=True)) * (y[start:stop + 1, 29:30] / np.sum(y[start:stop + 1, 28:31], axis=1, keepdims=True)) * (y[start:stop + 1, 34:35] / np.sum(y[start:stop + 1, 31:37], axis=1, keepdims=True)) * (y[start:stop + 1, 9:13] / np.sum(y[start:stop + 1, 9:13], axis=1, keepdims=True)),
                    # 1.2 + 2.2 + 3.2 + 4.1 + 10.2 + 11.5 + 5.1/.2/.3/.4
                    y[start:stop + 1, 6:7] * (y[start:stop + 1, 7:8] / np.sum(y[start:stop + 1, 7:9], axis=1, keepdims=True)) * (y[start:stop + 1, 29:30] / np.sum(y[start:stop + 1, 28:31], axis=1, keepdims=True)) * (y[start:stop + 1, 35:36] / np.sum(y[start:stop + 1, 31:37], axis=1, keepdims=True)) * (y[start:stop + 1, 9:13] / np.sum(y[start:stop + 1, 9:13], axis=1, keepdims=True)),
                    # 1.2 + 2.2 + 3.2 + 4.1 + 10.2 + 11.6 + 5.1/.2/.3/.4
                    y[start:stop + 1, 6:7] * (y[start:stop + 1, 7:8] / np.sum(y[start:stop + 1, 7:9], axis=1, keepdims=True)) * (y[start:stop + 1, 29:30] / np.sum(y[start:stop + 1, 28:31], axis=1, keepdims=True)) * (y[start:stop + 1, 36:37] / np.sum(y[start:stop + 1, 31:37], axis=1, keepdims=True)) * (y[start:stop + 1, 9:13] / np.sum(y[start:stop + 1, 9:13], axis=1, keepdims=True)),
                    # 1.2 + 2.2 + 3.2 + 4.1 + 10.3 + 11.1 + 5.1/.2/.3/.4
                    y[start:stop + 1, 6:7] * (y[start:stop + 1, 7:8] / np.sum(y[start:stop + 1, 7:9], axis=1, keepdims=True)) * (y[start:stop + 1, 30:31] / np.sum(y[start:stop + 1, 28:31], axis=1, keepdims=True)) * (y[start:stop + 1, 31:32] / np.sum(y[start:stop + 1, 31:37], axis=1, keepdims=True)) * (y[start:stop + 1, 9:13] / np.sum(y[start:stop + 1, 9:13], axis=1, keepdims=True)),
                    # 1.2 + 2.2 + 3.2 + 4.1 + 10.3 + 11.2 + 5.1/.2/.3/.4
                    y[start:stop + 1, 6:7] * (y[start:stop + 1, 7:8] / np.sum(y[start:stop + 1, 7:9], axis=1, keepdims=True)) * (y[start:stop + 1, 30:31] / np.sum(y[start:stop + 1, 28:31], axis=1, keepdims=True)) * (y[start:stop + 1, 32:33] / np.sum(y[start:stop + 1, 31:37], axis=1, keepdims=True)) * (y[start:stop + 1, 9:13] / np.sum(y[start:stop + 1, 9:13], axis=1, keepdims=True)),
                    # 1.2 + 2.2 + 3.2 + 4.1 + 10.3 + 11.3 + 5.1/.2/.3/.4
                    y[start:stop + 1, 6:7] * (y[start:stop + 1, 7:8] / np.sum(y[start:stop + 1, 7:9], axis=1, keepdims=True)) * (y[start:stop + 1, 30:31] / np.sum(y[start:stop + 1, 28:31], axis=1, keepdims=True)) * (y[start:stop + 1, 33:34] / np.sum(y[start:stop + 1, 31:37], axis=1, keepdims=True)) * (y[start:stop + 1, 9:13] / np.sum(y[start:stop + 1, 9:13], axis=1, keepdims=True)),
                    # 1.2 + 2.2 + 3.2 + 4.1 + 10.3 + 11.4 + 5.1/.2/.3/.4
                    y[start:stop + 1, 6:7] * (y[start:stop + 1, 7:8] / np.sum(y[start:stop + 1, 7:9], axis=1, keepdims=True)) * (y[start:stop + 1, 30:31] / np.sum(y[start:stop + 1, 28:31], axis=1, keepdims=True)) * (y[start:stop + 1, 34:35] / np.sum(y[start:stop + 1, 31:37], axis=1, keepdims=True)) * (y[start:stop + 1, 9:13] / np.sum(y[start:stop + 1, 9:13], axis=1, keepdims=True)),
                    # 1.2 + 2.2 + 3.2 + 4.1 + 10.3 + 11.5 + 5.1/.2/.3/.4
                    y[start:stop + 1, 6:7] * (y[start:stop + 1, 7:8] / np.sum(y[start:stop + 1, 7:9], axis=1, keepdims=True)) * (y[start:stop + 1, 30:31] / np.sum(y[start:stop + 1, 28:31], axis=1, keepdims=True)) * (y[start:stop + 1, 35:36] / np.sum(y[start:stop + 1, 31:37], axis=1, keepdims=True)) * (y[start:stop + 1, 9:13] / np.sum(y[start:stop + 1, 9:13], axis=1, keepdims=True)),
                    # 1.2 + 2.2 + 3.2 + 4.1 + 10.3 + 11.6 + 5.1/.2/.3/.4
                    y[start:stop + 1, 6:7] * (y[start:stop + 1, 7:8] / np.sum(y[start:stop + 1, 7:9], axis=1, keepdims=True)) * (y[start:stop + 1, 30:31] / np.sum(y[start:stop + 1, 28:31], axis=1, keepdims=True)) * (y[start:stop + 1, 36:37] / np.sum(y[start:stop + 1, 31:37], axis=1, keepdims=True)) * (y[start:stop + 1, 9:13] / np.sum(y[start:stop + 1, 9:13], axis=1, keepdims=True)),
                    # 1.2 + 2.2 + 3.2 + 4.2 + 5.1/.2/.3/.4
                    y[start:stop + 1, 6:7] * (y[start:stop + 1, 8:9] / np.sum(y[start:stop + 1, 7:9], axis=1, keepdims=True)) * (y[start:stop + 1, 9:13] / np.sum(y[start:stop + 1, 9:13], axis=1, keepdims=True)),
                    # 1.3
                    y[start:stop + 1, 2:3]
                ))
                y[np.isnan(y)] = 0
                y = np.repeat(y, 2, axis=0)

        view = [int(x / self.scale_factor) for x in self.patch_size]
        view.append(3)
        view_converter = DefaultViewConverter(view, axes)

        super(GZData, self).__init__(X=trainx, y=y, view_converter=view_converter)

        assert not np.any(np.isnan(self.X))


def get_data(tot=False):
    train_path = DATA_DIR+'gz_preprocessed_train' + str(SUBMODEL) + '.pkl'
    valid_path = DATA_DIR+'gz_preprocessed_valid' + str(SUBMODEL) + '.pkl'
    tottrain_path = DATA_DIR+'gz_preprocessed_tottrain' + str(SUBMODEL) + '.pkl'
    test_path = DATA_DIR+'gz_preprocessed_test' + str(SUBMODEL) + '.pkl'

    if os.path.exists(train_path) and os.path.exists(valid_path) and os.path.exists(test_path):

        print 'loading preprocessed data'
        datasets = OrderedDict()
        datasets['train'] = serial.load(train_path)
        datasets['valid'] = serial.load(valid_path)
        if tot:
            datasets['tottrain'] = serial.load(tottrain_path)
        datasets['test'] = serial.load(test_path)
        if tot:
            return datasets['tottrain'], datasets['valid'], datasets['test']
        else:
            return datasets['train'], datasets['valid'], datasets['test']
    else:
        print 'loading raw data...'
        data = GalaxyZoo.gzdeepdata.GZData(which_set='training', start=0, stop=39999)

        print 'preprocessing data...'
        pipeline = preprocessing.Pipeline()
        pipeline.items.append(preprocessing.GlobalContrastNormalization(use_std=True))
        pipeline.items.append(preprocessing.ZCA())

        print 'traindata'
        data.apply_preprocessor(preprocessor=pipeline, can_fit=True)
        # this path can be used for visualizing weights after training is done
        data.yaml_src = '!pkl: "%s"' % data
        # save
        data.use_design_loc(DATA_DIR+'train_design' + str(SUBMODEL) + '.npy')
        serial.save(DATA_DIR+'gz_preprocessed_train'+str(SUBMODEL) + '.pkl', data)

        print 'validdata'
        data = GalaxyZoo.gzdeepdata.GZData(which_set='training', start=40000, stop=61577)
        data.apply_preprocessor(preprocessor=pipeline, can_fit=False)
        # this path can be used for visualizing weights after training is done
        data.yaml_src = '!pkl: "%s"' % data
        # save
        data.use_design_loc(DATA_DIR+'valid_design' + str(SUBMODEL) + '.npy')
        serial.save(DATA_DIR+'gz_preprocessed_valid'+str(SUBMODEL) + '.pkl', data)

        print 'testdata'
        data = GalaxyZoo.gzdeepdata.GZData(which_set='test')
        data.apply_preprocessor(preprocessor=pipeline, can_fit=False)
        # this path can be used for visualizing weights after training is done
        data.yaml_src = '!pkl: "%s"' % data
        # save
        data.use_design_loc(DATA_DIR+'test_design' + str(SUBMODEL) + '.npy')
        serial.save(DATA_DIR+'gz_preprocessed_test'+str(SUBMODEL) + '.pkl', data)

        print 'tottraindata'
        data = GalaxyZoo.gzdeepdata.GZData(which_set='training')
        data.apply_preprocessor(preprocessor=pipeline, can_fit=True)
        # this path can be used for visualizing weights after training is done
        data.yaml_src = '!pkl: "%s"' % data
        # save
        data.use_design_loc(DATA_DIR+'tottrain_design' + str(SUBMODEL) + '.npy')
        serial.save(DATA_DIR+'gz_preprocessed_tottrain'+str(SUBMODEL) + '.pkl', data)

        print 'Finished, now re-run'
        return None, None, None

if __name__ == '__main__':
    pass
