'''@file read_audio
contains method to read the audio'''

import os
from StringIO import StringIO
import subprocess
import scipy.io.wavfile as wav

def read_audio(wavfile):
    '''
    read a wav file

    Args:
        wavfile: either a path to a wav file or a command to read and pipe
            an audio file

    Returns:
        - the sampling rate
        - the utterance as a numpy array
    '''

    if os.path.exists(wavfile):
        #its a file
        (rate, utterance) = wav.read(wavfile)
    elif wavfile[-1] == '|':
        #its a command

        #read the audio file
        pid = subprocess.Popen(wavfile + ' tee', shell=True,
                               stdout=subprocess.PIPE)
        output, _ = pid.communicate()
        output_buffer = StringIO(output)
        (rate, utterance) = wav.read(output_buffer)
    else:
        raise Exception('unexpected wavfile format %s' % wavfile)


    return rate, utterance
