'''@file structure
contains the Structure class'''

import xml.etree.ElementTree as ET
from assist.tasks.argtypes.factory import factory

class Structure(object):
    '''A task structure.

    A task structure contains tasks with named and typed arguments'''

    def __init__(self, struct_file):
        '''Structure constructor, reads the structure file and builds the
        task structure.

        Args:
            struct_file: the structure file, this is an xml file with the
                folowing structure:
                    <structure>
                        <types>
                            <typename supertype="<supertype>">
                                ...
                            </typename>
                            ...
                        <tasks>
                            <taskname argname="typename" .../>
                            ...
                        </tasks>
        '''

        #parse the struct file
        parsed = ET.parse(struct_file)

        #get the root of the parsed file, which should be structure
        structure_element = parsed.getroot()

        #get the children of the structure element which should be the types
        #element and the tasks element
        types_element = structure_element.find('types')
        tasks_element = structure_element.find('tasks')

        #process the types
        self.types = dict()
        for type_element in types_element:
            typename = type_element.tag
            if typename in self.types:
                raise Exception('typename %s defined twice in %s' %
                                (typename, struct_file))
            self.types[typename] = factory(type_element)

        #process the tasks
        self.tasks = dict()
        for task_element in tasks_element:
            taskname = task_element.tag
            if taskname in self.tasks:
                raise Exception('taskname %s defined twice in %s' %
                                (taskname, struct_file))
            self.tasks[taskname] = task_element.attrib
