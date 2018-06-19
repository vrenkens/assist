'''@file initializers.py
a file containing variable initializers for tensorflow'''

import math
import tensorflow as tf

class VoteTranformInitializer(tf.keras.initializers.Initializer):
    '''An Initializer for the voting transormation matrix in a capsule layer'''

    def __init__(self,
                 scale=1.0,
                 mode='fan_in',
                 distribution='normal',
                 seed=None,
                 dtype=tf.float32):
        '''
        Constructor

        args:
            scale: how to scale the initial values (default: 1.0)
            mode: One of 'fan_in', 'fan_out', 'fan_avg'. (default: fan_in)
            distribution: one of 'uniform' or 'normal' (default: normal)
            seed: A Python integer. Used to create random seeds.
            dtype: The data type. Only floating point types are supported.
        '''

        if scale <= 0.:
            raise ValueError('`scale` must be positive float.')
        if mode not in {'fan_in', 'fan_out', 'fan_avg'}:
            raise ValueError('Invalid `mode` argument:', mode)
        distribution = distribution.lower()
        if distribution not in {'normal', 'uniform'}:
            raise ValueError('Invalid `distribution` argument:', distribution)


        self.scale = scale
        self.mode = mode
        self.distribution = distribution
        self.seed = seed
        self.dtype = tf.as_dtype(dtype)

    def __call__(self, shape, dtype=None, partition_info=None):
        '''initialize the variables

        args:
            shape: List of `int` representing the shape of the output `Tensor`.
                [num_capsules_in, capsule_dim_in, num_capsules_out,
                 capsule_dim_out]
            dtype: (Optional) Type of the output `Tensor`.
            partition_info: (Optional) variable_scope._PartitionInfo object
                holding additional information about how the variable is
                partitioned. May be `None` if the variable is not partitioned.

        Returns:
            A `Tensor` of type `dtype` and `shape`.
        '''

        if dtype is None:
            dtype = self.dtype
        scale = self.scale
        scale_shape = shape

        if partition_info is not None:
            scale_shape = partition_info.full_shape

        if len(scale_shape) != 4:
            raise ValueError('expected shape to be of length 4', scale_shape)

        fan_in = scale_shape[1]
        fan_out = scale_shape[3]

        if self.mode == 'fan_in':
            scale /= max(1., fan_in)
        elif self.mode == 'fan_out':
            scale /= max(1., fan_out)
        else:
            scale /= max(1., (fan_in + fan_out) / 2.)

        if self.distribution == 'normal':
            stddev = math.sqrt(scale)
            return tf.truncated_normal(
                shape, 0.0, stddev, dtype, seed=self.seed)
        else:
            limit = math.sqrt(3.0 * scale)
            return tf.random_uniform(
                shape, -limit, limit, dtype, seed=self.seed)

    def get_config(self):
        '''get the initializer config'''

        return {
            'scale': self.scale,
            'mode': self.mode,
            'distribution': self.distribution,
            'seed': self.seed,
            'dtype': self.dtype.name
        }

def capsule_initializer(scale=1.0, seed=None, dtype=tf.float32):
    '''a VoteTranformInitializer'''

    return VoteTranformInitializer(
        scale=scale,
        mode='fan_avg',
        distribution='uniform',
        seed=seed,
        dtype=dtype
    )
