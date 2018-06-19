'''@file ops.py
contains some tensorflow ops'''

import tensorflow as tf

def squash(s, axis=-1, epsilon=1e-7, name=None):
    '''squash function'''
    with tf.name_scope(name, default_name="squash"):
        squared_norm = tf.reduce_sum(tf.square(s), axis=axis,
                                     keepdims=True)
        sn = tf.sqrt(squared_norm + epsilon)
        squash_factor = squared_norm / (1. + squared_norm)
        squash_factor /= sn

        return squash_factor * s

def splice(inputs, width, stride, axis=1, concat_axis=2):
    '''splice the inputs'''

    with tf.name_scope('splice'):

        div = width/2
        rem = width%2

        r = len(inputs.shape)
        inputs = tf.transpose(inputs, [axis]+range(axis)+range(axis+1, r))
        spliced = [inputs[::stride]]
        for i in range(1, div+rem):
            left = tf.pad(inputs[i:], [[0, i]] + [[0, 0]]*(r-1))
            left = left[::stride]
            right = tf.pad(inputs[:-i], [[i, 0]]  + [[0, 0]]*(r-1))
            right = right[::stride]
            spliced = [left] + spliced + [right]
        if not rem:
            i = div
            right = tf.pad(inputs[:-i], [[i, 0]]  + [[0, 0]]*(r-1))
            right = right[::stride]
            spliced = spliced + [right]

        spliced = [
            tf.transpose(s, range(1, axis+1)+[0]+range(axis+1, r))
            for s in spliced]

        spliced = tf.concat(spliced, concat_axis)

    return spliced

def conv_softmax(logits, width, stride, name=None):
    '''a probability function for convolutional capsules

    args:
        logits: batch_size x time x capsules_in*width x capsules_out
        width: the left and right window width

    returns:
        the weights same shape as logits
    '''

    with tf.name_scope(name, default_name='conv_softmax'):

        #interleave the initial logits with blocks
        with tf.name_scope('stride'):
            added = [tf.ones_like(logits)*logits.dtype.min]*(stride-1)
            logits = tf.stack([logits] + added, 2)
            logits = tf.reshape(
                logits,
                [logits.shape[0], -1, logits.shape[3], logits.shape[4]])

        logits = tf.split(logits, width, 2)

        div = width/2
        rem = width%2

        for i in range(1, div+rem):
            logits[div+rem-1-i] = tf.pad(
                logits[div+rem-1-i][:, :-i],
                [[0, 0], [i, 0], [0, 0], [0, 0]],
                constant_values=logits[0].dtype.min)
            logits[div+rem-1+i] = tf.pad(
                logits[div+rem-1+i][:, i:],
                [[0, 0], [0, i], [0, 0], [0, 0]],
                constant_values=logits[0].dtype.min)
        if not rem:
            i = div+rem
            logits[div+rem-1+i] = tf.pad(
                logits[div+rem-1+i][:, i:],
                [[0, 0], [0, i], [0, 0], [0, 0]],
                constant_values=logits[0].dtype.min)

        logits = tf.concat(logits, 3)
        weights = tf.nn.softmax(logits)
        weights = tf.split(weights, width, 3)

        for i in range(1, div+rem):
            weights[div+rem-1-i] = tf.pad(
                weights[div+rem-1-i][:, i:],
                [[0, 0], [0, i], [0, 0], [0, 0]])
            weights[div+rem-1+i] = tf.pad(
                weights[div+rem-1+i][:, :-i],
                [[0, 0], [i, 0], [0, 0], [0, 0]])
        if not rem:
            i = div+rem
            weights[div+rem-1+i] = tf.pad(
                weights[div+rem-1+i][:, :-i],
                [[0, 0], [i, 0], [0, 0], [0, 0]])

        weights = tf.concat(weights, 2)
        weights = weights[:, ::stride]

    return weights


def safe_norm(s, axis=-1, keepdims=False, epsilon=1e-7, name=None):
    '''compute a safe norm'''

    with tf.name_scope(name, default_name='safe_norm'):
        x = tf.reduce_sum(tf.square(s), axis=axis, keepdims=keepdims)
        return tf.sqrt(x + epsilon)
