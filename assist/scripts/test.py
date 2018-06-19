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
from assist.tasks import read_task
from assist.acquisition import model_factory
from assist.experiment import score

def main(expdir):
    '''main function'''

    #check if this experiment has been completed
    if os.path.exists(os.path.join(expdir, 'f1')):
        print 'result found %s' % expdir
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

    print 'loading model'
    model.load(os.path.join(expdir, 'model'))

    print 'prepping testing data'

    #load the testing features
    features = dict()
    for line in open(os.path.join(expdir, 'testfeats')):
        splitline = line.strip().split(' ')
        featsfile = ' '.join(splitline[1:])
        features[splitline[0]] = np.load(featsfile)

    #read the testtasks
    references = dict()
    for line in open(os.path.join(expdir, 'testtasks')):
        splitline = line.strip().split(' ')
        references[splitline[0]] = read_task.read_task(' '.join(splitline[1:]))

    print 'testing the model'

    #decode the test uterances
    decoded = model.decode(features)

    #write the decoded tasks to disc
    with open(os.path.join(expdir, 'dectasks'), 'w') as fid:
        for name, task in decoded.items():
            fid.write('%s %s\n' % (name, read_task.to_string(task)))

    (precision, recal, f1, macroprec, macrorecall, macrof1), scores = \
        score.score(decoded, references)

    print 'precision: %f' % precision
    print 'recal: %f' % recal
    print 'f1: %f' % f1
    print 'macro precision: %f' % macroprec
    print 'macro recal: %f' % macrorecall
    print 'macro f1: %f' % macrof1

    with open(os.path.join(expdir, 'precision'), 'w') as fid:
        fid.write(str(precision))
    with open(os.path.join(expdir, 'recal'), 'w') as fid:
        fid.write(str(recal))
    with open(os.path.join(expdir, 'f1'), 'w') as fid:
        fid.write(str(f1))
    with open(os.path.join(expdir, 'macroprecision'), 'w') as fid:
        fid.write(str(macroprec))
    with open(os.path.join(expdir, 'macrorecal'), 'w') as fid:
        fid.write(str(macrorecall))
    with open(os.path.join(expdir, 'macrof1'), 'w') as fid:
        fid.write(str(macrof1))

    score.write_scores(scores, expdir)

if __name__ == "__main__":

    #create the arguments parser
    parser = argparse.ArgumentParser()
    parser.add_argument('expdir')
    args = parser.parse_args()

    main(args.expdir)
