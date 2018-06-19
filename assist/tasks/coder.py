'''@file coder.py
contains the Coder class'''

import os
from abc import ABCMeta, abstractmethod
from assist.tools.tools import default_conf

class Coder(object):
    ''' Task coder

    Encodes task representations into a vector and estimates the
    task representation from a vector of probabilities'''

    __metaclass__ = ABCMeta

    def __init__(self, structure, conf):
        '''Coder constructor

        Args:
            structure: a Structure object
        '''

        #default conf file
        default = os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            'defaults',
            type(self).__name__.lower() + '.cfg')

        #apply the defaults
        if os.path.exists(default):
            default_conf(conf, default)

        self.structure = structure
        self.conf = dict(conf.items('coder'))

    @abstractmethod
    def encode(self, task):
        '''encode the task representation into a vector

        Args:
            task: the task reresentation as a Task object

        Returns:
            the encoded task representation as a numpy array
        '''

    @abstractmethod
    def decode(self, vector, cost):
        '''get the most likely task representation for the vector

        Args:
            vector: the vector to decode
            cost: a callable: cost(hypothesis, vector) that returns a cost for
                a hypothesis
        Returns:
            a task representation'''
