'''@file feature_computer.py
contains the FeatureComputer class'''

import os
from functools import partial
from abc import ABCMeta, abstractmethod
import numpy as np
from scipy.ndimage.filters import maximum_filter
import sigproc
import base
from assist.tools.tools import default_conf

class FeatureComputer(object):
    '''A featurecomputer is used to compute features'''

    __metaclass__ = ABCMeta

    def __init__(self, conf):
        '''
        FeatureComputer constructor

        Args:
            conf: the feature configuration
        '''

        #default conf file
        default = os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            'defaults',
            type(self).__name__.lower() + '.cfg')

        #apply the defaults
        if os.path.exists(default):
            default_conf(conf, default)

        self.conf = dict(conf.items('features'))

    def __call__(self, sig, rate):
        '''
        compute the features

        Args:
            sig: the audio signal as a 1-D numpy array
            rate: the sampling rate

        Returns:
            the features as a [seq_length x feature_dim] numpy array
        '''

        #snip the edges
        sig = sigproc.snip(sig, rate, float(self.conf['winlen']),
                           float(self.conf['winstep']))

        #compute the features
        feats = self.comp_feat(sig, rate)

        #apply vad
        if self.conf['vad'] == 'True':
            speechframes = vad(sig, rate, float(self.conf['winlen']),
                               float(self.conf['winstep']))
            feats = feats[speechframes, :]

        return feats

    @abstractmethod
    def comp_feat(self, sig, rate):
        '''
        compute the features

        Args:
            sig: the audio signal as a 1-D numpy array
            rate: the sampling rate

        Returns:
            the features as a [seq_length x feature_dim] numpy array
        '''

    @abstractmethod
    def get_dim(self):
        '''the feature dimemsion'''


def vad(sig, rate, winlen, winstep):
    '''do voice activity detection

    args:
        sig: the input signal as a numpy array
        rate: the sampling rate
        winlen: the window length
        winstep: the window step

    Returns:
        a numpy array of indices containing speech frames
    '''

    #apply preemphasis
    sig = sigproc.preemphasis(sig, 0.97)

    #do windowing windowing
    frames = sigproc.framesig(sig, winlen*rate, winstep*rate)

    #compute the squared frames and center them around zero mean
    sqframes = np.square(frames)
    sqframes = sqframes - sqframes.mean(1, keepdims=True)

    #compute the cross correlation between the frames and their square
    corr = np.array(map(partial(np.correlate, mode='same'), frames, sqframes))

    #compute the mel power spectrum of the correlated signal
    corrfft = np.fft.rfft(corr, 512)
    fb = base.get_filterbanks(26, 512, rate, 0, rate/2)
    E = np.absolute(np.square(corrfft).dot(fb.T))

    #do noise sniffing at the front and the back and select the lowest energy
    Efront = E[:20, :].mean(0)
    Eback = E[-20:, :].mean(0)
    if Efront.sum() < Eback.sum():
        Enoise = Efront
    else:
        Enoise = Eback

    #at every interval compute the mean ratio between the maximal energy in that
    #interval and the noise energy
    width = 12

    #apply max pooling to the energy
    Emax = maximum_filter(E, size=[width, 1], mode='constant')

    #compute the ratio between the smoothed energy and the noise energy
    ratio = np.log((Emax/Enoise).mean(axis=1))
    ratio = ratio/np.max(ratio)

    speechframes = np.where(ratio > 0.2)[0]

    return speechframes
