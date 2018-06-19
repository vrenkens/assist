'''@file model_factory.py
contains the model factory'''

def factory(name):
    '''model factory method

    args:
        name: tye type of model as a string

    Returns:
        a model class'''

    if name == 'rccn':
        import tfmodel.rccn
        return tfmodel.rccn.RCCN
    elif name == 'pccn':
        import tfmodel.pccn
        return tfmodel.pccn.PCCN
    elif name == 'encoder_decoder':
        import tfmodel.encoder_decoder
        return tfmodel.encoder_decoder.EncoderDecoder
    else:
        raise Exception('unknown acquisition type %s' % name)
