'''@file prepare_dataprep.py
do all the preparations for the data preparation'''

import os
import sys
sys.path.append(os.getcwd())
import shutil
import argparse
from ConfigParser import ConfigParser
import dataprep
from assist.tools import tools

def main(expdir, recipe, computing):
    '''main function'''

    if not os.path.isdir(expdir):
        os.makedirs(expdir)

    #read the database configuration
    if not os.path.exists(os.path.join(recipe, 'database.cfg')):
        raise Exception('cannot find database.cfg in %s' % recipe)

    dataconf = ConfigParser()
    dataconf.read(os.path.join(recipe, 'database.cfg'))

    shutil.copyfile(os.path.join(recipe, 'features.cfg'),
                    os.path.join(expdir, 'features.cfg'))

    for speaker in dataconf.sections():

        print '%s data preparation' % speaker

        #create the experiments directory
        if not os.path.isdir(os.path.join(expdir, speaker)):
            os.makedirs(os.path.join(expdir, speaker))

        #create a database config for this speaker
        speakerconf = ConfigParser()
        speakerconf.add_section('database')
        for option, value in dict(dataconf.items(speaker)).items():
            speakerconf.set('database', option, value)
        with open(os.path.join(expdir, speaker, 'database.cfg'), 'w') as fid:
            speakerconf.write(fid)

        #put a link to the feature conf
        tools.symlink(
            os.path.join(expdir, 'features.cfg'),
            os.path.join(expdir, speaker, 'features.cfg'))

        if computing in ('condor', 'condor_gpu'):
            #create the outputs directory
            if not os.path.isdir(os.path.join(expdir, speaker, 'outputs')):
                os.makedirs(os.path.join(expdir, speaker, 'outputs'))

            if computing == 'condor_gpu':
                jobfile = 'run_script_GPU.job'
            else:
                jobfile = 'run_script.job'

            #submit the condor job
            os.system(
                'condor_submit expdir=%s script=dataprep'
                ' assist/condor/%s' % (os.path.join(expdir, speaker), jobfile))
        else:
            dataprep.main(os.path.join(expdir, speaker))



if __name__ == '__main__':

    #parse the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('expdir', help='the experiments directory')
    parser.add_argument('recipe', help='the recipe directory')
    parser.add_argument('--computing', '-c',
                        help='the kind of computing you want to do')
    args = parser.parse_args()

    if args.computing and args.computing.strip() not in \
            ('condor', 'condor_gpu', 'local'):
        raise Exception('unknown computing mode %s' % args.computing)

    main(args.expdir, args.recipe, args.computing or 'local')
