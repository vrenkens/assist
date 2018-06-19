'''@file get_results.py
contains the get_results script'''

import os

def get_results(expdir, toread):
    '''read the results for an experiment on a database'''

    if not os.path.isdir(expdir):
        raise Exception('cannot find expdir: %s' % expdir)

    #the results per speaker
    results = dict()

    #get a list of speaker directories
    speakers = [s for s in os.listdir(expdir)
                if os.path.isdir(os.path.join(expdir, s))]

    skipped = 0

    for speaker in speakers:
        results_speaker = dict()

        exps = [e for e in os.listdir(os.path.join(expdir, speaker))
                if os.path.isdir(os.path.join(expdir, speaker, e))]

        for exp in exps:

            exppath = os.path.join(expdir, speaker, exp)
            expname = '-'.join([speaker, exp])

            if not os.path.isfile(os.path.join(exppath, toread)):
                print 'no result in %s' % exppath
                skipped += 1
                continue

            #read the result
            with open(os.path.join(exppath, toread)) as fid:
                result = float(fid.read())

            #get the amount of training data
            numex = 0
            for _ in open(os.path.join(expdir, speaker, exp, 'trainfeats')):
                numex += 1

            results_speaker[expname] = (numex, result)

        if results_speaker:
            results[speaker] = results_speaker

    return results
