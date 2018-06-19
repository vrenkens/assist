'''@file layers.py
contains tensorflow layers'''

from __future__ import division
import tensorflow as tf
import ops
from initializers import capsule_initializer

class Capsule(tf.layers.Layer):
    '''a capsule layer'''

    def __init__(
            self, num_capsules, capsule_dim,
            kernel_initializer=None,
            logits_initializer=None,
            routing_iters=3,
            activation_fn=None,
            probability_fn=None,
            activity_regularizer=None,
            trainable=True,
            name=None,
            **kwargs):

        '''Capsule layer constructor

        args:
            num_capsules: number of output capsules
            capsule_dim: output capsule dimsension
            kernel_initializer: an initializer for the prediction kernel
            logits_initializer: the initializer for the initial logits
            routing_iters: the number of routing iterations (default: 5)
            activation_fn: a callable activation function (default: squash)
            probability_fn: a callable that takes in logits and returns weights
                (default: tf.nn.softmax)
            activity_regularizer: Regularizer instance for the output (callable)
            trainable: wether layer is trainable
            name: the name of the layer
        '''

        super(Capsule, self).__init__(
            trainable=trainable,
            name=name,
            activity_regularizer=activity_regularizer,
            **kwargs)

        self.num_capsules = num_capsules
        self.capsule_dim = capsule_dim
        self.kernel_initializer = kernel_initializer or capsule_initializer()
        self.logits_initializer = logits_initializer or tf.zeros_initializer()
        self.routing_iters = routing_iters
        self.activation_fn = activation_fn or ops.squash
        self.probability_fn = probability_fn or tf.nn.softmax

    def build(self, input_shape):
        '''creates the variables of this layer

        args:
            input_shape: the shape of the input
        '''

        #pylint: disable=W0201

        #input dimensions
        num_capsules_in = input_shape[-2].value
        capsule_dim_in = input_shape[-1].value

        if num_capsules_in is None:
            raise ValueError('number of input capsules must be defined')
        if capsule_dim_in is None:
            raise ValueError('input capsules dimension must be defined')

        self.kernel = self.add_variable(
            name='kernel',
            dtype=self.dtype,
            shape=[num_capsules_in, capsule_dim_in,
                   self.num_capsules, self.capsule_dim],
            initializer=self.kernel_initializer)

        self.logits = self.add_variable(
            name='init_logits',
            dtype=self.dtype,
            shape=[num_capsules_in, self.num_capsules],
            initializer=self.logits_initializer
        )

        super(Capsule, self).build(input_shape)

    #pylint: disable=W0221
    def call(self, inputs):
        '''
        apply the layer

        args:
            inputs: the inputs to the layer. the final two dimensions are
                num_capsules_in and capsule_dim_in

        returns the output capsules with the last two dimensions
            num_capsules and capsule_dim
        '''

        #compute the predictions
        predictions, logits = self.predict(inputs)

        #cluster the predictions
        outputs = self.cluster(predictions, logits)

        return outputs

    def predict(self, inputs):
        '''
        compute the predictions for the output capsules and initialize the
        routing logits

        args:
            inputs: the inputs to the layer. the final two dimensions are
                num_capsules_in and capsule_dim_in

        returns: the output capsule predictions
        '''

        with tf.name_scope('predict'):

            #number of shared dimensions
            rank = len(inputs.shape)
            shared = rank-2

            #put the input capsules as the first dimension
            inputs = tf.transpose(inputs, [shared] + range(shared) + [rank-1])

            #compute the predictins
            predictions = tf.map_fn(
                fn=lambda x: tf.tensordot(x[0], x[1], [[shared], [0]]),
                elems=(inputs, self.kernel),
                dtype=self.dtype or tf.float32)

            #transpose back
            predictions = tf.transpose(
                predictions, range(1, shared+1)+[0]+[rank-1, rank])

            logits = self.logits
            for i in range(shared):
                if predictions.shape[shared-i-1].value is None:
                    shape = tf.shape(predictions)[shared-i-1]
                else:
                    shape = predictions.shape[shared-i-1].value
                tile = [shape] + [1]*len(logits.shape)
                logits = tf.tile(tf.expand_dims(logits, 0), tile)

        return predictions, logits

    def cluster(self, predictions, logits):
        '''cluster the predictions into output capsules

        args:
            predictions: the predicted output capsules
            logits: the initial routing logits

        returns:
            the output capsules
        '''

        with tf.name_scope('cluster'):

            #define m-step
            def m_step(l):
                '''m step'''
                with tf.name_scope('m_step'):
                    #compute the capsule contents
                    w = self.probability_fn(l)
                    caps = tf.reduce_sum(
                        tf.expand_dims(w, -1)*predictions, -3)

                return caps, w

            #define body of the while loop
            def body(l):
                '''body'''

                caps, _ = m_step(l)
                caps = self.activation_fn(caps)

                #compare the capsule contents with the predictions
                similarity = tf.reduce_sum(
                    predictions*tf.expand_dims(caps, -3), -1)

                return l + similarity

            #get the final logits with the while loop
            lo = tf.while_loop(
                lambda l: True,
                body, [logits],
                maximum_iterations=self.routing_iters)

            #get the final output capsules
            capsules, _ = m_step(lo)
            capsules = self.activation_fn(capsules)

        return capsules

    def compute_output_shape(self, input_shape):
        '''compute the output shape'''

        if input_shape[-2].value is None:
            raise ValueError(
                'The number of capsules must be defined, but saw: %s'
                % input_shape)
        if input_shape[-1].value is None:
            raise ValueError(
                'The capsule dimension must be defined, but saw: %s'
                % input_shape)

        return input_shape[:-2].concatenate(
            [self.num_capsules, self.capsule_dim])


def capsule(
        inputs, num_capsules, capsule_dim,
        kernel_initializer=None,
        logits_initializer=None,
        routing_iters=3,
        activation_fn=None,
        probability_fn=None,
        activity_regularizer=None,
        trainable=True,
        name=None,
        reuse=None):

    '''apply capsule layer to inputs

    args:
        inputs: the input capsules
        num_capsules: number of output capsules
        capsule_dim: output capsule dimsension
        kernel_initializer: an initializer for the prediction kernel
        logits_initializer: the initializer for the initial logits
        routing_iters: the number of routing iterations (default: 5)
        activation_fn: a callable activation function (default: squash)
        probability_fn: a callable that takes in logits and returns weights
            (default: tf.nn.softmax)
        activity_regularizer: Regularizer instance for the output (callable)
        trainable: wether layer is trainable
        name: the name of the layer
        reuse: Boolean, whether to reuse the weights of a previous layer
            by the same name.

    returns:
        the output capsules
    '''

    layer = Capsule(
        num_capsules=num_capsules,
        capsule_dim=capsule_dim,
        kernel_initializer=kernel_initializer,
        logits_initializer=logits_initializer,
        routing_iters=routing_iters,
        activation_fn=activation_fn,
        probability_fn=probability_fn,
        activity_regularizer=activity_regularizer,
        trainable=trainable,
        name=name,
        dtype=inputs.dtype.base_dtype,
        _scope=name,
        _reuse=reuse)

    return layer(inputs)

class Conv1DCapsule(Capsule):
    '''a 1D convolutional capsule layer'''

    def __init__(
            self, num_capsules, capsule_dim, width, stride,
            kernel_initializer=None,
            logits_initializer=None,
            routing_iters=3,
            activation_fn=None,
            probability_fn=None,
            activity_regularizer=None,
            trainable=True,
            name=None,
            **kwargs):

        '''Capsule layer constructor

        args:
            num_capsules: number of output capsules
            capsule_dim: output capsule dimsension
            width: the width of the convolutional kernel
            stride: the convolutional stride
            kernel_initializer: an initializer for the prediction kernel
            logits_initializer: the initializer for the initial logits
            routing_iters: the number of routing iterations (default: 5)
            activation_fn: a callable activation function (default: squash)
            probability_fn: a callable that takes in logits and returns weights
                (default: tf.nn.convolutional_softmax)
            activity_regularizer: Regularizer instance for the output (callable)
            trainable: wether layer is trainable
            name: the name of the layer
        '''

        self.width = width
        self.stride = stride

        super(Conv1DCapsule, self).__init__(
            num_capsules=num_capsules,
            capsule_dim=capsule_dim,
            kernel_initializer=kernel_initializer,
            logits_initializer=logits_initializer,
            routing_iters=routing_iters,
            activation_fn=activation_fn,
            probability_fn=probability_fn,
            activity_regularizer=activity_regularizer,
            trainable=trainable,
            name=name,
            **kwargs
        )

    def predict(self, inputs):
        '''
        compute the predictions for the output capsules

        args:
            inputs: the inputs to the layer. the final two dimensions are
                num_capsules_in and capsule_dim_in

        returns: the output capsule predictions
        '''

        with tf.name_scope('predict'):

            #put the input capsules as the first dimension
            inputs = tf.transpose(inputs, [2, 0, 1, 3])

            #compute the predictins
            predictions = tf.map_fn(
                fn=lambda x: tf.nn.conv1d(
                    value=x[0],
                    filters=x[1],
                    padding='VALID',
                    stride=self.stride),
                elems=(inputs, self.kernel),
                dtype=self.dtype or tf.float32)

            #transpose back
            predictions = tf.transpose(predictions, [1, 2, 0, 3])

            #reshape the last dimension
            predictions = tf.reshape(predictions, [
                predictions.shape[0].value,
                tf.shape(predictions)[1],
                predictions.shape[2].value,
                self.num_capsules,
                self.capsule_dim
            ])

            logits = self.logits
            logits = tf.tile(
                logits[tf.newaxis, tf.newaxis, :, :],
                [predictions.shape[0].value, tf.shape(predictions)[1], 1, 1])

        return predictions, logits

    def build(self, input_shape):
        '''creates the variables of this layer

        args:
            input_shape: the shape of the input
        '''


        #pylint: disable=W0201

        #input dimensions
        num_capsules_in = input_shape[-2].value
        capsule_dim_in = input_shape[-1].value

        if num_capsules_in is None:
            raise ValueError('number of input capsules must be defined')
        if capsule_dim_in is None:
            raise ValueError('input capsules dimension must be defined')

        self.kernel = self.add_variable(
            name='kernel',
            dtype=self.dtype,
            shape=[num_capsules_in, self.width*capsule_dim_in,
                   self.num_capsules, self.capsule_dim],
            initializer=self.kernel_initializer)
        self.kernel = tf.reshape(
            self.kernel,
            [num_capsules_in, self.width, capsule_dim_in,
             self.num_capsules*self.capsule_dim])

        self.logits = self.add_variable(
            name='init_logits',
            dtype=self.dtype,
            shape=[num_capsules_in, self.num_capsules],
            initializer=self.logits_initializer
        )

        self.built = True


def conv1d_capsule(
        inputs, num_capsules, capsule_dim, width, stride,
        kernel_initializer=None,
        logits_initializer=None,
        routing_iters=5,
        activation_fn=None,
        probability_fn=None,
        activity_regularizer=None,
        trainable=True,
        name=None,
        reuse=None):

    '''apply capsule layer to inputs

    args:
        inputs: the input capsules
        num_capsules: number of output capsules
        capsule_dim: output capsule dimsension
        width: the width of the convolutional kernel
        stride: the convolutional stride
        kernel_initializer: an initializer for the prediction kernel
        logits_initializer: the initializer for the initial logits
        routing_iters: the number of routing iterations (default: 5)
        activation_fn: a callable activation function (default: squash)
        probability_fn: a callable that takes in logits and returns weights
            (default: tf.nn.softmax)
        activity_regularizer: Regularizer instance for the output (callable)
        trainable: wether layer is trainable
        name: the name of the layer
        reuse: Boolean, whether to reuse the weights of a previous layer
            by the same name.

    returns:
        the output capsules
    '''

    layer = Conv1DCapsule(
        num_capsules=num_capsules,
        capsule_dim=capsule_dim,
        width=width,
        stride=stride,
        kernel_initializer=kernel_initializer,
        logits_initializer=logits_initializer,
        routing_iters=routing_iters,
        activation_fn=activation_fn,
        probability_fn=probability_fn,
        activity_regularizer=activity_regularizer,
        trainable=trainable,
        name=name,
        dtype=inputs.dtype.base_dtype,
        _scope=name,
        _reuse=reuse)

    return layer(inputs)

class TCRCCapsule(Capsule):
    '''a capsule layer to go from time coded capsules to rate coded capsules'''

    def predict(self, inputs):
        '''
        compute the predictions for the output capsules

        args:
            inputs: the inputs to the layer. the final two dimensions are
                num_capsules_in and capsule_dim_in

        returns: the output capsule predictions and initial logits
        '''

        predictions, logits = super(TCRCCapsule, self).predict(inputs)

        predictions = tf.concat(tf.unstack(predictions, axis=-3), 1)
        logits = tf.concat(tf.unstack(logits, axis=-2), 1)

        return predictions, logits

def tcrc_capsule(
        inputs, num_capsules, capsule_dim,
        kernel_initializer=None,
        logits_initializer=None,
        routing_iters=5,
        activation_fn=None,
        probability_fn=None,
        activity_regularizer=None,
        trainable=True,
        name=None,
        reuse=None):

    '''apply capsule layer to inputs

    args:
        inputs: the input capsules
        num_capsules: number of output capsules
        capsule_dim: output capsule dimsension
        kernel_initializer: an initializer for the prediction kernel
        logits_initializer: the initializer for the initial logits
        routing_iters: the number of routing iterations (default: 5)
        activation_fn: a callable activation function (default: squash)
        probability_fn: a callable that takes in logits and returns weights
            (default: tf.nn.softmax)
        activity_regularizer: Regularizer instance for the output (callable)
        trainable: wether layer is trainable
        name: the name of the layer
        reuse: Boolean, whether to reuse the weights of a previous layer
            by the same name.

    returns:
        the output capsules
    '''

    layer = TCRCCapsule(
        num_capsules=num_capsules,
        capsule_dim=capsule_dim,
        kernel_initializer=kernel_initializer,
        logits_initializer=logits_initializer,
        routing_iters=routing_iters,
        activation_fn=activation_fn,
        probability_fn=probability_fn,
        activity_regularizer=activity_regularizer,
        trainable=trainable,
        name=name,
        dtype=inputs.dtype.base_dtype,
        _scope=name,
        _reuse=reuse)

    return layer(inputs)
