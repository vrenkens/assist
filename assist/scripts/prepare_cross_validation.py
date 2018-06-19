'''@file prepare_cross_validation.py
do all the preparations for cross validation'''

import os
import sys
sys.path.append(os.getcwd())
import shutil
import argparse
import cPickle as pickle
from ConfigParser import ConfigParser
import random
import itertools
import numpy as np
from assist.tasks.structure import Structure
from assist.tasks import coder_factory
from assist.tasks.read_task import read_task
from assist.experiment.make_blocks import make_blocks
import train_test
from assist.tools import tools

def main(expdir, recipe, computing):
    '''main function'''

    overwrite = False
    if os.path.isdir(expdir):
        text = ''
        while text not in ('o', 'r'):
            text = raw_input('%s already exists, do you want to '
                             'resume experiment (r) or overwrite (o) '
                             '(respond with o or r)' % expdir)
        if text == 'o':
            overwrite = True

    else:
        #create the experiments directory
        os.makedirs(expdir)

    #copy the config files
    if overwrite:
        shutil.copyfile(os.path.join(recipe, 'acquisition.cfg'),
                        os.path.join(expdir, 'acquisition.cfg'))
    else:
        tools.safecopy(os.path.join(recipe, 'acquisition.cfg'),
                       os.path.join(expdir, 'acquisition.cfg'))

    shutil.copyfile(os.path.join(recipe, 'coder.cfg'),
                    os.path.join(expdir, 'coder.cfg'))
    shutil.copyfile(os.path.join(recipe, 'structure.xml'),
                    os.path.join(expdir, 'structure.xml'))

    #read the cross_validation config file
    expconf = ConfigParser()
    expconf.read(os.path.join(recipe, 'cross_validation.cfg'))

    #default conf file
    default = os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        'defaults',
        'cross_validation.cfg')

    #apply the defaults
    if os.path.exists(default):
        tools.default_conf(expconf, default)

    expconf = dict(expconf.items('cross_validation'))

    #read the data config file
    if not os.path.exists(os.path.join(recipe, 'database.cfg')):
        raise Exception('cannot find database.cfg in %s' % recipe)

    dataconf = ConfigParser()
    dataconf.read(os.path.join(recipe, 'database.cfg'))

    #read the coder config file
    coderconf = ConfigParser()
    coderconf.read(os.path.join(expdir, 'coder.cfg'))

    for speaker in dataconf.sections():

        print 'speaker: %s' % (speaker)

        #create the speaker directory
        if os.path.isdir(os.path.join(expdir, speaker)):
            if overwrite:
                shutil.rmtree(os.path.join(expdir, speaker))
                os.makedirs(os.path.join(expdir, speaker))
        else:
            os.makedirs(os.path.join(expdir, speaker))

        #create a task structure file
        structure = Structure(os.path.join(expdir, 'structure.xml'))

        #create a coder
        coder = coder_factory.factory(coderconf.get('coder', 'name'))(
            structure, coderconf)

        #read and code all the tasks
        labelvecs = []
        names = []
        taskstrings = dict()
        for line in open(dataconf.get(speaker, 'tasks')):
            splitline = line.strip().split(' ')
            name = splitline[0]
            names.append(name)
            taskstring = ' '.join(splitline[1:])
            taskstrings[name] = taskstring
            task = read_task(taskstring)
            labelvecs.append(coder.encode(task))

        #devide the data into blocks
        blocksfile = os.path.join(expdir, speaker, 'blocks.pkl')
        if os.path.exists(blocksfile):
            with open(blocksfile, 'rb') as fid:
                blocks = pickle.load(fid)
        else:
            blocks = make_blocks(np.array(labelvecs), expconf,
                                 dataconf.get(speaker, 'features'))
            with open(blocksfile, 'wb') as fid:
                pickle.dump(blocks, fid)

        #create train-testsets for all experiments

        #seed the random number generator
        random.seed(3105)
        trainids = [None]*(len(blocks)-1)
        testids = [None]*(len(blocks)-1)
        for b in range(len(blocks)-1):
            trainids[b] = [None]*int(expconf['numexp'])
            testids[b] = [None]*int(expconf['numexp'])
            for e in range(int(expconf['numexp'])):
                trainids[b][e] = list(
                    itertools.chain.from_iterable(random.sample(blocks, b+1)))
                testids[b][e] = [x for x in range(len(names))
                                 if x not in trainids[b][e]]

        #read the feature files
        features = dict()
        for l in open(os.path.join(dataconf.get(speaker, 'features'), 'feats')):
            splitline = l.strip().split(' ')
            features[splitline[0]] = ' '.join(splitline[1:])

        #create an expdir for each experiment
        b = int(expconf['startblocks']) - 1

        while True:
            for e in range(int(expconf['numexp'])):

                print 'train blocks: %d, experiment %s' % (b+1, e)

                #creat the directory
                subexpdir = os.path.join(expdir, speaker,
                                         '%dblocks_exp%d' % (b+1, e))

                if os.path.exists(os.path.join(subexpdir, 'f1')):
                    continue

                if not os.path.isdir(subexpdir):
                    os.makedirs(subexpdir)

                #create pointers to the config files
                tools.symlink(os.path.join(expdir, 'acquisition.cfg'),
                              os.path.join(subexpdir, 'acquisition.cfg'))
                tools.symlink(os.path.join(expdir, 'coder.cfg'),
                              os.path.join(subexpdir, 'coder.cfg'))
                tools.symlink(os.path.join(expdir, 'structure.xml'),
                              os.path.join(subexpdir, 'structure.xml'))

                if not os.path.exists(os.path.join(subexpdir, 'trainfeats')):
                    trainutts = [names[i] for i in trainids[b][e]]
                    print 'number of examples: %d' % len(trainutts)
                    testutts = [names[i] for i in testids[b][e]]

                    #create the train and test sets
                    tools.writefile(
                        os.path.join(subexpdir, 'trainfeats'),
                        {utt: features[utt] for utt in trainutts})
                    tools.writefile(
                        os.path.join(subexpdir, 'traintasks'),
                        {utt: taskstrings[utt] for utt in trainutts})
                    tools.writefile(
                        os.path.join(subexpdir, 'testfeats'),
                        {utt: features[utt] for utt in testutts})
                    tools.writefile(
                        os.path.join(subexpdir, 'testtasks'),
                        {utt: taskstrings[utt] for utt in testutts})

                if computing in ('condor', 'condor_gpu'):
                    #create the outputs directory
                    if not os.path.isdir(os.path.join(subexpdir, 'outputs')):
                        os.makedirs(os.path.join(subexpdir, 'outputs'))

                    if computing == 'condor_gpu':
                        jobfile = 'run_script_GPU.job'
                    else:
                        jobfile = 'run_script.job'

                    #only submit the job if it not running yet
                    in_queue = os.popen(
                        'if condor_q -nobatch -wide | grep -q %s; '
                        'then echo true; else echo false; fi' %
                        subexpdir).read().strip() == 'true'

                    #submit the condor job
                    if not in_queue:
                        os.system('condor_submit expdir=%s script=train_test'
                                  ' assist/condor/%s'
                                  % (subexpdir, jobfile))
                else:
                    train_test.main(subexpdir)

            newb = (b + 1)*int(expconf['scale']) + int(expconf['increment']) - 1
            newb = min(newb, len(blocks) - 2)
            if b == newb:
                break
            else:
                b = newb

if __name__ == '__main__':

    #parse the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('expdir', help='the experiments directory')
    parser.add_argument('recipe', help='the recipe directory')
    parser.add_argument('--computing', '-c',
                        help='the kind of computing you want to do')
    args = parser.parse_args()

    if args.computing and args.computing not in \
            ('condor', 'condor_gpu', 'local'):
        raise Exception('unknown computing mode %s' % args.computing)

    main(args.expdir, args.recipe, args.computing)
