'''@file distributor_capsules.py
Contains the DistributorCapsules class'''

import tfmodel
import layers
import ops
import tensorflow as tf

class RCCN(tfmodel.TFModel):
    '''an encoder-decoder with dynamic routing acquisition model'''

    def model(self, inputs, seq_length):
        '''apply the model'''

        with tf.variable_scope('model'):

            #encode the features
            encoded, seq_length = self.encoder(inputs, seq_length)

            #compute the primary capsules
            prim_capsules, contrib = self.primary_capsules(encoded, seq_length)

            #get the output_capsules
            output_capsules, alignment = self.output_capsules(
                prim_capsules, contrib)

            #compute the label probabilities
            if self.conf['slot_filling'] == 'True':
                probs, alignment = self.slot_filling(output_capsules, alignment)
            else:
                probs = ops.safe_norm(output_capsules)

            tf.add_to_collection('image', tf.expand_dims(alignment, 3, 'ali'))
            tf.add_to_collection('store', tf.identity(alignment, 'alignment'))
            tf.add_to_collection(
                'store', tf.identity(output_capsules, 'output_capsules'))

        return probs

    def loss(self, targets, probs):
        '''compute the loss

        args:
            targets: the reference targets
            probs: the label probabilities

        returns: the loss'''

        with tf.name_scope('compute_loss'):
            #compute the loss
            iw = float(self.conf['insertion_weight'])
            up = float(self.conf['upper_prob'])
            lp = float(self.conf['lower_prob'])
            iloss = iw*tf.reduce_mean(
                tf.reduce_sum((1-targets)*tf.maximum(probs-lp, 0)**2, 1))
            dloss = tf.reduce_mean(
                tf.reduce_sum(targets*tf.maximum(up-probs, 0)**2, 1))
            loss = dloss + iloss

        return loss

    def encoder(self, features, seq_length):
        '''encode the input features

        args:
            features: a [N x T x F] tensor
            seq_length: an [N] tensor containing the sequence lengths

        returns:
            - the encoded features
            - the encode features sequence lengths
        '''

        with tf.variable_scope('encoder'):

            encoded = tf.identity(features, 'features')
            seq_length = tf.identity(seq_length, 'input_seq_length')

            for l in range(int(self.conf['numlayers_encoder'])):
                with tf.variable_scope('layer%d' % l):
                    num_units = int(self.conf['numunits_encoder'])
                    fw = tf.contrib.rnn.GRUCell(num_units)
                    bw = tf.contrib.rnn.GRUCell(num_units)
                    encoded, _ = tf.nn.bidirectional_dynamic_rnn(
                        fw, bw, encoded, dtype=tf.float32,
                        sequence_length=seq_length)

                    encoded = tf.concat(encoded, 2)

                    if l != int(self.conf['numlayers_encoder']) - 1:
                        with tf.name_scope('sub-sample'):
                            encoded = encoded[:, ::int(self.conf['subsample'])]
                        seq_length = tf.to_int32(tf.ceil(
                            tf.to_float(seq_length)/
                            float(self.conf['subsample'])))

            encoded = tf.identity(encoded, 'encoded')
            seq_length = tf.identity(seq_length, 'output_seq_length')

        return encoded, seq_length

    def primary_capsules(self, encoded, seq_length):
        '''compute the primary capsules

        args:
            encoded: encoded sequences [batch_size x time x dim]
            seq_length: the sequence lengths [batch_size]

        returns:
            the primary capsules
                [batch_size x num_capsules x capsule_dim]
            the contribution of the time steps in the primary capsules
                [batch_size x time x num_capsules x capsule_dim]
        '''

        with tf.variable_scope('primary_capsules'):

            encoded = tf.identity(encoded, 'encoded')
            seq_length = tf.identity(seq_length, 'seq_length')

            num_capsules = int(self.conf['num_capsules'])
            capsule_dim = int(self.conf['capsule_dim'])

            #distribute timesteps over primary capsules
            distribution = tf.layers.dense(encoded, num_capsules, tf.nn.softmax)

            #put a weight on all the timesteps
            attention = tf.layers.dense(encoded, 1, tf.nn.sigmoid)
            weights = attention*distribution

            #compute the weighted averages
            combinations = tf.matmul(weights, encoded, transpose_a=True)

            #map the averages to primary capsule orientations
            layer = tf.layers.Dense(
                capsule_dim, use_bias=False,
                name='orientations')
            orientations = layer(combinations)

            primary_capsules = ops.squash(orientations)

            #get the squash factor
            squash = tf.get_default_graph().get_tensor_by_name(
                'model/primary_capsules/squash/div_1:0')
            contrib = layer(encoded)
            contrib = (
                tf.expand_dims(squash, 1)
                *tf.expand_dims(contrib, 2)
                *tf.expand_dims(weights, 3))

            tf.add_to_collection(
                'image', tf.expand_dims(weights, 3, 'prim_weights'))

            primary_capsules = tf.identity(primary_capsules, 'primary_capsules')

        return primary_capsules, contrib

    def output_capsules(self, rc_capsules, contrib):
        '''compute the output capsules

        args:
            rc_capsules: the rate coded capsules
                [batch_size x num_capsules x capsule_dim]
            contrib: the conttibution of each timestep in the rc capsules
                [batch_size x time x num_capsules x capsule_dim]

        returns:
            the output_capsules [batch_size x num_capsules x capsule_dim]
            the alignment of the output capsules to the timesteps
                [batch_size x time x num_capsules]
        '''

        with tf.variable_scope('output_capsules'):

            capsules = tf.identity(rc_capsules, 'primary_capsules')
            contrib = tf.identity(contrib, 'contrib')
            r = int(self.conf['capsule_ratio'])**int(self.conf['num_rc_layers'])
            num_capsules = self.num_output_capsules*r
            capsule_dim = int(self.conf['output_capsule_dim'])/r

            for l in range(int(self.conf['num_rc_layers'])):
                with tf.variable_scope('layer%d' % l):

                    num_capsules /= int(self.conf['capsule_ratio'])
                    capsule_dim *= int(self.conf['capsule_ratio'])

                    layer = layers.Capsule(
                        num_capsules=num_capsules,
                        capsule_dim=capsule_dim,
                        routing_iters=int(self.conf['routing_iters'])
                    )

                    capsules = layer(capsules)

                    #get the predictions for the contributions
                    contrib_predict, _ = layer.predict(contrib)

                    #get the final routing logits
                    logits = tf.get_default_graph().get_tensor_by_name(
                        layer.scope_name + '/cluster/while/Exit_1:0')

                    #get the final squash factor
                    sf = tf.get_default_graph().get_tensor_by_name(
                        layer.scope_name + '/cluster/squash/div_1:0')

                    #get the routing weight
                    weights = layer.probability_fn(logits)

                    weights *= tf.transpose(sf, [0, 2, 1])
                    weights = tf.expand_dims(tf.expand_dims(weights, 1), 4)

                    contrib = tf.reduce_sum(contrib_predict*weights, 2)

            alignment = tf.reduce_sum(
                contrib*tf.expand_dims(capsules, 1), 3,
                name='alignment')
            capsules = tf.identity(capsules, 'output_capsules')

        return capsules, alignment


    def slot_filling(self, output_capsules, alignments):
        '''assign the output capsules to the appropriate slot

        args:
            output_capsules: [batch_size x num_values x capsule_dim]
            alignments: the alignments of the output_capsules
                [batch_size x time x num_values]

        returns:
            the output label probabilities: [batch_size x num_labels]
        '''

        with tf.variable_scope('slot_filling'):

            valids = self.coder.valids
            ids = []
            probs = []
            alis = []
            #all_outputs = tf.concat(tf.unstack(output_capsules, axis=1), 1)

            for i, val in enumerate(valids):
                with tf.variable_scope(val):
                    capsule = output_capsules[:, i]
                    alignment = alignments[:, :, i]
                    alignment = tf.expand_dims(alignment, 2)
                    p = tf.layers.dense(
                        capsule,
                        len(valids[val]),
                        tf.nn.softmax,
                        name=val)
                    alignment *= tf.expand_dims(tf.square(p), 1)
                    p *= ops.safe_norm(capsule, keepdims=True)
                    probs.append(p)

                ids += valids[val].values()
                alis.append(alignment)

            probs = tf.concat(probs, 1)
            probs = tf.gather(probs, ids, axis=1)
            alignments = tf.concat(alis, 2)
            alignments = tf.gather(alignments, ids, axis=2)

        return probs, alignments

    @property
    def num_output_capsules(self):
        '''number of output capsules'''

        if self.conf['slot_filling'] == 'True':
            return len(self.coder.valids)
        else:
            return self.coder.numlabels
