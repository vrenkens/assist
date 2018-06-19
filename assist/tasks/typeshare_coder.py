'''@file typeshare_coder.py
contains the TypeShareCoder class'''

import numpy as np
from assist.tasks.read_task import Task
import coder

class TypeShareCoder(coder.Coder):
    ''' a Coder that shares the places for args with the same type'''

    def __init__(self, structure, conf):
        '''Coder constructor

        Args:
            structure: a Structure object
        '''

        #super constructor
        super(TypeShareCoder, self).__init__(structure, conf)

        #give an index to all the tasks
        self.taskindices = {t : i for i, t in enumerate(structure.tasks.keys())}

        #give an index to all the arguments
        self.typeindices = dict()
        index = len(structure.tasks)
        for argtype in structure.types:
            self.typeindices[argtype] = {
                t : i + index
                for i, t in enumerate(structure.types[argtype].options)
                }
            index += len(structure.types[argtype].options)

        #save the number of labels
        self.numlabels = index

    def encode(self, task):
        '''encode the task representation into a vector

        Args:
            task: the task reresentation as a Task object

        Returns:
            the encoded task representation as a numpy array
        '''

        #create the vector
        vector = np.zeros([self.numlabels])

        #check the correctness of the task representation
        if task.name not in self.structure.tasks:
            raise Exception('unknown task %s' % task.name)
        for arg in task.args:
            if arg not in self.structure.tasks[task.name]:
                raise Exception('unknown argument %s' % arg)
            argtype = self.structure.tasks[task.name][arg]
            if task.args[arg] not in self.structure.types[argtype].options:
                raise Exception('unknown option %s' % task.args[arg])

        #set the index of the task to one
        vector[self.taskindices[task.name]] = 1

        #put the argument indices to one
        for arg in task.args:
            argtype = self.structure.tasks[task.name][arg]
            vector[self.typeindices[argtype][task.args[arg]]] = 1

        return vector

    def decode(self, vector, cost):
        '''get the most likely task representation for the vector

        Args:
            vector: the vector to decode
            cost: a callable: cost(hypothesis, vector) that returns a cost for
                a hypothesis
        Returns:
            a task representation'''


        #threshold the vector
        threshold = min(float(self.conf['threshold']), np.max(vector))
        vector = np.where(vector >= threshold, vector, np.zeros_like(vector))

        best = (None, 0)
        for task in self.structure.tasks:

            args = {}
            for arg in self.structure.tasks[task]:
                argtype = self.structure.tasks[task][arg]
                argvec = vector[self.typeindices[argtype].values()]
                if not np.any(argvec):
                    continue
                argid = np.argmax(argvec)
                args[arg] = self.typeindices[argtype].keys()[argid]

            if not args and not vector[self.taskindices[task]]:
                continue

            c = cost(self.encode(Task(name=task, args=args)), vector)
            if best[0] is None or c < best[1]:
                best = (Task(name=task, args=args), c)

        return best[0]
