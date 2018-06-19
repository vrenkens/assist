'''@file compare_results.py
compare the results of experiments on the same database

usage: python plot_alignment.py expdir name
    expdir: the experiments directory
    result: the name of the utterance you want to plot
'''

import sys
import os
sys.path.append(os.getcwd())
from ConfigParser import ConfigParser
import numpy as np
import matplotlib.pyplot as plt
from assist.tasks import coder_factory
from assist.tasks.structure import Structure
from assist.tasks.read_task import read_task

def main(expdir, name):
    '''main function'''

    colorlist = ['black']
    linestyles = ['-', '--', ':', '-.']

    #read the alignment
    alignment = np.load(os.path.join(expdir, 'alignment', '%s.npy' % name))

    #read the decoded task
    taskstrings = dict()
    for line in open(os.path.join(expdir, 'dectasks')):
        splitline = line.strip().split(' ')
        taskstrings[splitline[0]] = ' '.join(splitline[1:])

    #read the coder config file
    coderconf = ConfigParser()
    coderconf.read(os.path.join(expdir, 'coder.cfg'))

    #create a task structure file
    structure = Structure(os.path.join(expdir, 'structure.xml'))

    #create a coder
    coder = coder_factory.factory(coderconf.get('coder', 'name'))(
        structure, coderconf)

    #encode the decoded task
    labelvec = coder.encode(read_task(taskstrings[name]))

    #create the legend
    legend = coder.labelids
    alignment = [alignment[:, l] for l in range(coder.numlabels) if labelvec[l]]
    legend = [legend[l] for l in range(coder.numlabels) if labelvec[l]]

    for i, ali in enumerate(alignment):
        plt.plot(
            ali,
            color=colorlist[i%len(colorlist)],
            linestyle=linestyles[i%len(linestyles)],
            label=legend[i])
    plt.legend()
    plt.show()



if __name__ == '__main__':

    main(sys.argv[1], sys.argv[2])
