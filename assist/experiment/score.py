'''@file score.py
contais the score method'''

from __future__ import division
import os

def score(decoded, references):
    '''score the performance

    args:
        decoded: the decoded tasks as a dictionary
        references: the reference tasks as a dictionary

    returns:
        - the scores as a tuple (precision, recal f1)
        - a dictionary with scores per label
    '''

    #the number of true positives
    correct = {}
    #the number of positives = true positives + false positives
    positives = {}
    #the number of references = true positives + false negatives
    labels = {}

    #count the number of correct arguments in the correct tasks
    for i, r in references.items():
        d = decoded[i]

        #update positives
        if d.name not in positives:
            positives[d.name] = [0, {}]
        positives[d.name][0] += 1
        for arg, val in d.args.items():
            if arg not in positives[d.name][1]:
                positives[d.name][1][arg] = {}
            if val not in positives[d.name][1][arg]:
                positives[d.name][1][arg][val] = 0
            positives[d.name][1][arg][val] += 1

        #update labels
        if r.name not in labels:
            labels[r.name] = [0, {}]
        labels[r.name][0] += 1
        for arg, val in r.args.items():
            if arg not in labels[r.name][1]:
                labels[r.name][1][arg] = {}
            if val not in labels[r.name][1][arg]:
                labels[r.name][1][arg][val] = 0
            labels[r.name][1][arg][val] += 1

        #update correct correct
        if r.name not in correct:
            correct[r.name] = [0, {}]
        if r.name == d.name:
            correct[r.name][0] += 1
            for arg, val in r.args.items():
                if arg not in correct[r.name][1]:
                    correct[r.name][1][arg] = {}
                if val not in correct[r.name][1][arg]:
                    correct[r.name][1][arg][val] = 0
                if arg in d.args and val == d.args[arg]:
                    correct[r.name][1][arg][val] += 1

    #collect the scores
    numpositives = 0
    numlabels = 0
    numcorrect = 0
    numitems = 0
    macroprec = 0
    macrorecall = 0
    macrof1 = 0
    scores = {}
    for t in labels:

        if t not in positives:
            positives[t] = [0, {}]

        #udate global scores
        numlabels += labels[t][0]
        numcorrect += correct[t][0]
        numpositives += positives[t][0]

        scores[t] = [comp_score(correct[t][0], labels[t][0], positives[t][0]),
                     {}]
        numitems += 1
        macroprec += scores[t][0][0]
        macrorecall += scores[t][0][1]
        macrof1 += scores[t][0][2]

        for arg in labels[t][1]:

            scores[t][1][arg] = {}

            if arg not in positives[t][1]:
                positives[t][1][arg] = {}
            if arg not in correct[t][1]:
                correct[t][1][arg] = {}

            for val in labels[t][1][arg]:
                if val not in positives[t][1][arg]:
                    positives[t][1][arg][val] = 0
                if val not in correct[t][1][arg]:
                    correct[t][1][arg][val] = 0

                #update global scores
                numlabels += labels[t][1][arg][val]
                numcorrect += correct[t][1][arg][val]
                numpositives += positives[t][1][arg][val]

                scores[t][1][arg][val] = comp_score(
                    correct[t][1][arg][val],
                    labels[t][1][arg][val],
                    positives[t][1][arg][val])

                numitems += 1
                macroprec += scores[t][1][arg][val][0]
                macrorecall += scores[t][1][arg][val][1]
                macrof1 += scores[t][1][arg][val][2]

    s = comp_score(numcorrect, numlabels, numpositives)
    if numitems:
        macroprec /= numitems
        macrorecall /= numitems
        macrof1 /= numitems

    return (s[0], s[1], s[2], macroprec, macrorecall, macrof1), scores

def write_scores(scores, location):
    '''write the scores to a readable file'''

    fid = open(os.path.join(location, 'label_f1'), 'w')
    rid = open(os.path.join(location, 'label_recal'), 'w')
    pid = open(os.path.join(location, 'label_precision'), 'w')
    lid = open(os.path.join(location, 'label_labelcount'), 'w')
    oid = open(os.path.join(location, 'label_positives'), 'w')
    tid = open(os.path.join(location, 'label_truepositives'), 'w')

    fid.write('Label f1 scores\n')
    rid.write('Label recal\n')
    pid.write('Label precision\n')
    lid.write('Label reference positive count\n')
    oid.write('Label detected positive count\n')
    tid.write('Label true positive count\n')

    write_index(scores, 0, '%f', pid)
    write_index(scores, 1, '%f', rid)
    write_index(scores, 2, '%f', fid)
    write_index(scores, 3, '%d', lid)
    write_index(scores, 4, '%d', oid)
    write_index(scores, 5, '%d', tid)

    fid.close()
    rid.close()
    pid.close()
    lid.close()
    oid.close()
    tid.close()

def write_index(scores, index, fmt, fid):
    '''write a part of the scores'''

    for t in scores:
        fid.write(('\n%s: ' + fmt) % (t, scores[t][0][index]))
        for arg in scores[t][1]:
            fid.write('\n\t%s:' % arg)
            for val in scores[t][1][arg]:
                fid.write(
                    ('\n\t\t%s: ' + fmt) %
                    (val, scores[t][1][arg][val][index]))

def comp_score(correct, labels, positives):
    '''compute scores'''

    if labels:
        recal = correct/labels
    else:
        recal = 1

    if positives:
        precision = correct/positives
    else:
        precision = 1

    if precision + recal:
        f1 = 2*precision*recal/(precision + recal)
    else:
        f1 = 0

    return precision, recal, f1, labels, positives, correct
