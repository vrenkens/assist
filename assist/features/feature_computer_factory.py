'''@file feature_computer_factory.py
contains the FeatureComputer factory'''

from . import mfcc, mfcc_pitch, fbank

def factory(feature):
    '''
    create a FeatureComputer

    Args:
        feature: the feature computer type
    '''

    if feature == 'fbank':
        return fbank.Fbank
    elif feature == 'mfcc':
        return mfcc.Mfcc
    elif feature == 'mfcc_pitch':
        return mfcc_pitch.Mfcc_pitch
    else:
        raise Exception('Undefined feature type: %s' % feature)
