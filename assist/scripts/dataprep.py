'''@file dataprep.py
this script do the data preparation
'''

import os
import sys
sys.path.append(os.getcwd())
import argparse
from ConfigParser import ConfigParser
import numpy as np
from assist.features import feature_computer_factory
from assist.features.read_audio import read_audio

def main(expdir):
    '''main function'''

    featureconf_file = os.path.join(expdir, 'features.cfg')
    dataconf_file = os.path.join(expdir, 'database.cfg')

    #read the data config file
    dataconf = ConfigParser()
    dataconf.read(dataconf_file)
    dataconf = dict(dataconf.items('database'))

    #read the features config file
    featconf = ConfigParser()
    featconf.read(featureconf_file)

    #create the feature computer
    feature_computer = feature_computer_factory.factory(
        featconf.get('features', 'name'))(featconf)

    #compute the features for all the audio files in the database and store them
    #on disk
    if not os.path.isdir(dataconf['features']):
        os.makedirs(dataconf['features'])

    with open(os.path.join(dataconf['features'], 'feats'), 'w') as fid:
        for line in open(dataconf['audio']):
            splitline = line.strip().split(' ')
            name = splitline[0]
            print 'computing features for %s' % name
            wavfile = ' '.join(splitline[1:])
            rate, sig = read_audio(wavfile)
            if len(sig.shape) == 2:
                # feature computers assume mono
                sig = np.int16(np.mean(sig, axis=1))
            feats = feature_computer(sig, rate)
            filename = os.path.join(dataconf['features'], name + '.npy')
            fid.write(name + ' ' + filename + '\n')
            np.save(filename, feats)

if __name__ == "__main__":

    #create the arguments parser
    parser = argparse.ArgumentParser()
    parser.add_argument('expdir')
    args = parser.parse_args()

    main(args.expdir)
