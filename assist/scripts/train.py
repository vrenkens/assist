'''@file train_test.py
do training followed by testing
'''

import os
import sys
sys.path.append(os.getcwd())
import argparse
from ConfigParser import ConfigParser
import numpy as np
from assist.tasks.structure import Structure
from assist.tasks import coder_factory
from assist.acquisition import model_factory

def main(expdir):
    '''main function'''

    #check if this experiment has been completed
    if os.path.isdir(os.path.join(expdir, 'model')):
        return

    #read the acquisition config file
    acquisitionconf = ConfigParser()
    acquisitionconf.read(os.path.join(expdir, 'acquisition.cfg'))

    #read the coder config file
    coderconf = ConfigParser()
    coderconf.read(os.path.join(expdir, 'coder.cfg'))

    #create a task structure file
    structure = Structure(os.path.join(expdir, 'structure.xml'))

    #create a coder
    coder = coder_factory.factory(coderconf.get('coder', 'name'))(
        structure, coderconf)

    #create an acquisition model
    model = model_factory.factory(acquisitionconf.get('acquisition', 'name'))(
        acquisitionconf, coder, expdir)

    print 'prepping training data'

    #load the training features
    features = dict()
    for line in open(os.path.join(expdir, 'trainfeats')):
        splitline = line.strip().split(' ')
        featsfile = ' '.join(splitline[1:])
        features[splitline[0]] = np.load(featsfile)

    #read the traintasks
    taskstrings = dict()
    for line in open(os.path.join(expdir, 'traintasks')):
        splitline = line.strip().split(' ')
        taskstrings[splitline[0]] = ' '.join(splitline[1:])

    #create lists of features and training tasks
    examples = {utt: (features[utt], taskstrings[utt]) for utt in taskstrings}

    print 'training acquisition model'
    model.train(examples)

    #save the trained model
    model.save(os.path.join(expdir, 'model'))

if __name__ == "__main__":

    #create the arguments parser
    parser = argparse.ArgumentParser()
    parser.add_argument('expdir')
    args = parser.parse_args()

    main(args.expdir)
