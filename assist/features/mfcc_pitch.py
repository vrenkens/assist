'''@file mfcc_pitch.py
contains the mfcc feature computer'''

import numpy as np
import base
import feature_computer
import scipy.io.wavfile as wav
import os
import sys

class Mfcc_pitch(feature_computer.FeatureComputer):
    '''the feature computer class to compute MFCC_PITCH features'''

    def comp_feat(self, sig, rate):
        '''
        compute the features

        Args:
            sig: the audio signal as a 1-D numpy array
            rate: the sampling rate

        Returns:
            the features as a [seq_length x feature_dim] numpy array
        '''

        feat, energy = base.mfcc(sig, rate, self.conf)

        # write the wav to temporary location and invoke external pitch extractor.
        # make sure 'reaper' is in your $PATH
        tempdir = os.path.join('/tmp',str(os.getpid()))
        if not os.path.isdir(tempdir):
            os.makedirs(tempdir)
        name = 'mix'
        wav.write(os.path.join(tempdir, name + '.wav'),rate,np.int16(sig))
        os.system('reaper -i ' + os.path.join(tempdir, name + '.wav') + ' -f '+ os.path.join(tempdir, name + '.txt -a -e 0.01'))
        pitch = np.loadtxt(os.path.join(tempdir, name + '.txt'),skiprows=7)[:,2]
        pitch = np.pad(pitch,(0,max(0,feat.shape[0]-pitch.shape[0])),'edge')
        # linear interpolation in voiceless regions
        voiceless = np.where(pitch==-1)[0]
        jump = np.where((voiceless[1:]-voiceless[:-1])>1)[0]
        segments = np.split(voiceless,jump+1)
        for seg in segments:
            idx1 = seg[0]-1
            idx2 = seg[-1]+1
            val1 = -1
            val2 = -1
            if idx1>=0:
                val1 = pitch[idx1]
            if idx2<pitch.size:
                val2 = pitch[idx2]
            if val1 == -1: #segment starts at utterence start
                val1 = val2
            if val2 == -1: # segment ends at utterance end
                val2 = val1
            if val1 == -1: #segment is the whole utterance => make up a value
                val1 = 150
                val2 = 150
            #interpolate
            pitch[seg] = (val2-val1) * (np.array(seg)-idx1)/float(idx2-idx1) + val1

        feat = np.append(feat, pitch[:feat.shape[0], np.newaxis], 1)


        if self.conf['include_energy'] == 'True':
            feat = np.append(feat, energy[:, np.newaxis], 1)

        if self.conf['dynamic'] == 'delta':
            feat = base.delta(feat)
        elif self.conf['dynamic'] == 'ddelta':
            feat = base.ddelta(feat)
        elif self.conf['dynamic'] != 'nodelta':
            raise Exception('unknown dynamic type')

        #mean and variance normalize the features
        if self.conf['mvn'] == 'True':
            feat = (feat - feat.mean(0))/(feat.std(0)+1e-20)  # features could be constant, e.g. voiceless speech

        return feat

    def get_dim(self):
        '''the feature dimemsion'''

        dim = int(self.conf['numcep'])+1

        if self.conf['include_energy'] == 'True':
            dim += 1

        if self.conf['dynamic'] == 'delta':
            dim *= 2
        elif self.conf['dynamic'] == 'ddelta':
            dim *= 3

        return dim
