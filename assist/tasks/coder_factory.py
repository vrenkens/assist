'''@file coder_factory.py
contains the coder factory'''

from . import typeshare_coder, typesplit_coder

def factory(name):
    '''create a Coder object

    args:
        name: the name of the coder

    returns:
        a Coder class
    '''

    if name == 'typeshare_coder':
        return typeshare_coder.TypeShareCoder
    elif name == 'typesplit_coder':
        return typesplit_coder.TypeSplitCoder
    else:
        raise Exception('unknown coder %s' % name)
