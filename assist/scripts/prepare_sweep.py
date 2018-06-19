'''@file sweep.py
do a parameter sweep'''

import os
import sys
sys.path.append(os.getcwd())
import argparse
from ConfigParser import ConfigParser
import shutil
from subprocess import Popen, PIPE

def main(expdir, recipe, command, sweepfile, computing):
    '''main function'''

    text = ''
    if os.path.isdir(expdir):
        while text not in ('o', 'r'):
            text = raw_input('%s already exists, do you want to '
                             'resume experiment (r) or overwrite (o) '
                             '(respond with o or r)' % expdir)

    else:
        #create the experiments directory
        text = 'r'
        os.makedirs(expdir)

    sweep = ConfigParser()
    sweep.read(sweepfile)

    recipes = os.path.join(expdir, 'recipes')
    if os.path.isdir(recipes):
        shutil.rmtree(recipes)
    os.makedirs(recipes)

    for section in sweep.sections():
        subexpdir = os.path.join(expdir, section)
        if not os.path.isdir(subexpdir):
            os.makedirs(subexpdir)
        recipedir = os.path.join(recipes, section)
        shutil.copytree(recipe, recipedir)
        for name, value in sweep.items(section):
            conf, sec, field = name.split('.')
            config = os.path.join(recipedir, conf + '.cfg')
            modify_value(config, sec, field, value)

        p = Popen(
            args=['run', command, subexpdir, recipedir, '-c', computing],
            stdin=PIPE)
        p.communicate(input=text)
        returncode = p.wait()
        if returncode:
            return returncode




def modify_value(config, section, field, value):
    '''modify a value in a config file'''

    #read the config
    conf = ConfigParser()
    conf.read(config)

    #modify the value
    conf.set(section, field, value)

    #write the config
    with open(config, 'w') as fid:
        conf.write(fid)

if __name__ == '__main__':

    #parse the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('expdir', help='the experiments directory')
    parser.add_argument('recipe', help='the recipe directory')
    parser.add_argument('command', help='the command to be run')
    parser.add_argument('sweepfile', help='the file containing the parameters')
    parser.add_argument('--computing', '-c',
                        help='the kind of computing you want to do')
    args = parser.parse_args()

    comp = args.computing or 'local'
    if comp not in ('condor', 'condor_gpu', 'local'):
        raise Exception('unknown computing mode %s' % comp)

    if args.command not in ('cross_validation', 'train', 'test'):
        raise Exception('unknown command %s' % args.command)

    main(args.expdir, args.recipe, args.command, args.sweepfile, comp)
