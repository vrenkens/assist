'''@file plot_database.py
plot the result for a single database

usage: python plot_database expdir [-s --speakers 1] [-t -types 1]
    expdir: the experiments directory
    -s --speakers: if all speakers should be plotted seperately (default 0)
    -t -types: if all types should be plotted or only f1-score (default 0)
    '''

import os
import sys
sys.path.append(os.getcwd())
import argparse
import matplotlib.pyplot as plt
from statsmodels.nonparametric.smoothers_lowess import lowess
import numpy as np
from get_results import get_results

def main(expdir, plot_speakers, plot_types):
    '''main function'''

    #lowess parameters
    smooth = lambda y, x: lowess(
        y, x + 1e-12 * np.random.randn(len(x)),
        frac=1.0/3,
        it=0,
        delta=1.0,
        return_sorted=True)

    typestyle = dict()
    typestyle['f1'] = '-'
    typestyle['precision'] = '--'
    typestyle['recal'] = ':'

    results = dict()
    results['f1'] = get_results(expdir, 'f1')
    if plot_types:
        results['precision'] = get_results(expdir, 'precision')
        results['recal'] = get_results(expdir, 'recal')
    speakers = results['f1'].keys()

    #do LOWESS smooting per speaker
    numexamples = {s: np.array(r.values())[:, 0] for s, r in results['f1'].items()}
    results = {t: {s: np.array(r.values())[:, 1] for s, r in results[t].items()}
               for t in results}

    #lowess fit all the results for all the speakers speakers
    fits = {t: {s: smooth(a, numexamples[s]) for s, a in r.items()}
            for t, r in results.items()}

    if plot_speakers:
        #plot all the speakers
        for s in speakers:
            plt.figure(s)
            for t in fits:
                if t != 'f1' and not plot_types:
                    continue
                plt.plot(
                    fits[t][s][:, 0], fits[t][s][:, 1],
                    label=t)
            plt.scatter(numexamples[s], results['f1'][s])
            plt.legend()
            plt.xlabel('# Examples')
            plt.ylabel('Accuracy')

    #concatenate all the results
    cnumex = np.concatenate(numexamples.values())
    conc = {t: np.concatenate(r.values()) for t, r in results.items()}

    #fit all the results
    cfits = {t: smooth(conc[t], cnumex) for t in conc}

    #plot averages
    plt.figure('averages')
    for t in cfits:
        if t != 'f1' and not plot_types:
            continue
        plt.plot(
            cfits[t][:, 0], cfits[t][:, 1],
            color='black',
            linestyle=typestyle[t],
            linewidth=2,
            label=t)
    for s in speakers:
        plt.plot(
            fits['f1'][s][:, 0], fits['f1'][s][:, 1],
            label=s)

    plt.legend()
    plt.xlabel('# Examples')
    plt.ylabel('Accuracy')

    plt.show()



if __name__ == '__main__':

    #parse the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('expdir', help='directory containing the experiments')
    parser.add_argument('--speakers', '-s',
                        help='if all speakers should be plotted seperately')
    parser.add_argument('--types', '-t',
                        help='if all types should be plotted or only f1-score')
    args = parser.parse_args()

    main(args.expdir, args.speakers, args.types)
