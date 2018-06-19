'''@file model.py
Contains the Model class'''

import os
from abc import ABCMeta, abstractmethod
from assist.tools.tools import default_conf

class Model(object):
    '''General speech acquisition model class'''

    __metaclass__ = ABCMeta

    def __init__(self, conf, coder, expdir):
        '''model constructor

        Args:
            conf: the model configuration as as dictionary of strings
            coder: an object that encodes the tasks
            expdir: the experiments directory
        '''

        #default conf file
        default = os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            'defaults',
            type(self).__name__.lower() + '.cfg')

        #apply the defaults
        if os.path.exists(default):
            default_conf(conf, default)

        self.conf = dict(conf.items('acquisition'))
        self.coder = coder
        self.expdir = expdir

    @abstractmethod
    def train(self, examples):
        '''train the model

        Args:
            examples: the training examples as a dict of pairs containing the
                inputs and reference tasks
        '''

    @abstractmethod
    def decode(self, inputs):
        '''decode using the model

        Args:
            inputs: the inputs as a dict

        Returns:
            the estimated task representation as a dict
        '''

    @abstractmethod
    def load(self, directory):
        '''load the model

        Args:
            directory: the directory where the model was saved
        '''

    @abstractmethod
    def save(self, directory):
        '''save the model

        Args:
            directory: the directory where the model should be saved
        '''
