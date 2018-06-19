'''@file read_semantic.py
contains the read_semantic function'''

import xml.etree.ElementTree as ET
from collections import namedtuple

Task = namedtuple('Task', ['name', 'args'])

def read_task(string):
    '''read a task string

    Args:
        string: an xml string with the following structure:
            <taskname argname=arg ...>
    Returns:
        the task representation'''

    #parse the file and get the root
    element = ET.fromstring(string)
    args = element.attrib
    args = {arg:args[arg] for arg in args if args[arg] != ''}
    return Task(name=element.tag, args=args)

def to_string(task):
    '''convert task to string'''

    root = ET.Element(task.name, attrib=task.args)
    return ET.tostring(root)
