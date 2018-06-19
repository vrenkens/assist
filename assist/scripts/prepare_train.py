'''@file prepare_train.py
do all the preparations for training'''

import os
import sys
sys.path.append(os.getcwd())
import shutil
import argparse
from ConfigParser import ConfigParser
import train

def main(expdir, recipe, computing):
    '''main function'''

    if os.path.isdir(expdir):
        text = ''
        while text not in ('y', 'n'):
            text = raw_input('%s allready exists, do you want to overwite '
                             '(the directory will be removed)? '
                             '(y or n)' % expdir)
        if text == 'n':
            return 0
        shutil.rmtree(os.path.join(expdir))

    os.makedirs(expdir)

    #copy the config files
    shutil.copyfile(os.path.join(recipe, 'acquisition.cfg'),
                    os.path.join(expdir, 'acquisition.cfg'))
    shutil.copyfile(os.path.join(recipe, 'coder.cfg'),
                    os.path.join(expdir, 'coder.cfg'))
    shutil.copyfile(os.path.join(recipe, 'structure.xml'),
                    os.path.join(expdir, 'structure.xml'))

    #read the training config file
    trainconf = ConfigParser()
    trainconf.read(os.path.join(recipe, 'train.cfg'))

    #read the data config file
    if not os.path.exists(os.path.join(recipe, 'database.cfg')):
        raise Exception('cannot find database.cfg in %s' % recipe)

    dataconf = ConfigParser()
    dataconf.read(os.path.join(recipe, 'database.cfg'))

    #create the training files
    fid = open(os.path.join(expdir, 'trainfeats'), 'w')
    tid = open(os.path.join(expdir, 'traintasks'), 'w')

    for s in trainconf.get('train', 'datasections').split(' '):
        with open(os.path.join(dataconf.get(s, 'features'), 'feats')) as f:
            lines = f.readlines()
            lines = [s + '_' + line for line in lines]
            fid.writelines(lines)
        with open(dataconf.get(s, 'tasks')) as f:
            lines = f.readlines()
            lines = [s + '_' + line for line in lines]
            tid.writelines(lines)

    fid.close()
    tid.close()

    if computing in ('condor', 'condor_gpu'):
        #create the outputs directory
        if not os.path.isdir(os.path.join(expdir, 'outputs')):
            os.makedirs(os.path.join(expdir, 'outputs'))

        if computing == 'condor_gpu':
            jobfile = 'run_script_GPU.job'
        else:
            jobfile = 'run_script.job'

        #only submit the job if it not running yet
        in_queue = os.popen(
            'if condor_q -nobatch -wide | grep -q %s; '
            'then echo true; else echo false; fi' %
            expdir).read().strip() == 'true'

        #submit the condor job
        if not in_queue:
            os.system('condor_submit expdir=%s script=train'
                      ' assist/condor/%s'
                      % (expdir, jobfile))
    else:
        train.main(expdir)

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
