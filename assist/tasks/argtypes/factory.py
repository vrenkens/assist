'''@file argtype_factory.py
contains the type factory'''

from assist.tasks.argtypes.argtypes import Enumerable

def factory(type_element):
    '''creates a type based on the Tree element

    Args:
        type_element: the type element as a Element from an ElementTree

    Returns:
        a argument type
    '''

    if type_element.attrib['supertype'] == 'enumerable':
        options = type_element.text.split('\n')
        options = [option.strip() for option in options
                   if option.strip() != '']
        argtype = Enumerable(options=options)
    else:
        raise Exception('unknown argtype %s' % type_element.supertype)

    return argtype
