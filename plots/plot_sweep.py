'''@file plot_sweep.py
compare the results of experiments on the same database

usage: python compare_results.py result sweepfile expdir
    result: what you want to plot (e.g. f1)
    sweepfile: the parameter sweep file
    expdir: the experiments directory of one of the experiments
'''

import sys
import os
import itertools
import numpy as np
from ConfigParser import ConfigParser
from statsmodels.nonparametric.smoothers_lowess import lowess
import matplotlib.pyplot as plt
from get_results import get_results

def main(result, sweepfile, expdir):
    '''main function'''

    colorlist = ['red', 'blue', 'cyan', 'green', 'yellow', 'magenta',
                 'purple', 'pink', 'gold', 'navy', 'olive', 'grey']
    linestyles = ['-']

    #read the sweepfile
    sweep = ConfigParser()
    sweep.read(sweepfile)

    #colorlist = ['black']
    #linestyles = ['-', '--', ':', '-.']

    #lowess parameters
    smooth = lambda y, x: lowess(
        y, x + 1e-12 * np.random.randn(len(x)),
        frac=2.0/3,
        it=0,
        delta=1.0,
        return_sorted=True)
    plot_speakers = True

    #read all the results
    results = [
        get_results(os.path.join(expdir, section), result)
        for section in sweep.sections()]
    expnames = sweep.sections()

    if plot_speakers:
        for speaker in results[0]:
            plt.figure(speaker)
            for i, result in enumerate(results):
                if speaker not in result:
                    continue
                sort = np.array(result[speaker])
                sort = sort[np.argsort(sort[:, 0], axis=0), :]
                fit = smooth(sort[:, 1], sort[:, 0])
                plt.plot(fit[:, 0], fit[:, 1],
                         color=colorlist[i%len(colorlist)],
                         linestyle=linestyles[i%len(linestyles)],
                         label=expnames[i])
            plt.legend(loc='lower right')
            plt.xlabel('# Examples')
            plt.ylabel('Accuracy')

    #concatenate all the results
    concatenated = [
        np.array(list(itertools.chain.from_iterable(result.values())))
        for result in results]

    #sort the concatenated data
    sort = [
        c[np.argsort(c[:, 0], axis=0), :]
        if c.size else None for c in concatenated]

    #smooth all the results
    fit = [smooth(s[:, 1], s[:, 0]) if s is not None else None for s in sort]

    plt.figure('result')
    for i, f in enumerate(fit):
        if f is None:
            continue
        plt.plot(f[:, 0], f[:, 1],
                 color=colorlist[i%len(colorlist)],
                 linestyle=linestyles[i%len(linestyles)],
                 label=expnames[i])

    plt.legend(loc='lower right')
    plt.xlabel('# Examples')
    plt.ylabel('Accuracy')
    plt.show()


if __name__ == '__main__':

    main(sys.argv[1], sys.argv[2], sys.argv[3])
