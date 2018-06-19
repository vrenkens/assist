'''@file full_capsules.py
Contains the FullCapsules class'''

import tfmodel
import layers
import ops
import tensorflow as tf

class PCCN(tfmodel.TFModel):
    '''an encoder-decoder with dynamic routing acquisition model'''

    def model(self, inputs, seq_length):
        '''apply the model'''

        with tf.variable_scope('model'):

            #encode the features
            encoded, seq_length = self.encoder(inputs, seq_length)

            #compute the primary capsules
            prim_capsules = self.primary_capsules(encoded, seq_length)

            #compute time coded capsules
            tc_capsules, seq_length = self.tc_capsules(
                prim_capsules, seq_length)

            #get the rate coded capsules
            rc_capsules, contrib = self.rc_capsules(tc_capsules)

            #get the output_capsules
            output_capsules, alignment = self.output_capsules(
                rc_capsules, contrib)

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
                [batch_size x time x num_capsules x capsule_dim]
        '''

        with tf.variable_scope('primary_capsules'):

            encoded = tf.identity(encoded, 'encoded')
            seq_length = tf.identity(seq_length, 'seq_length')

            r = int(self.conf['capsule_ratio'])**int(self.conf['num_tc_layers'])
            num_capsules = int(self.conf['num_tc_capsules'])*r
            capsule_dim = int(self.conf['tc_capsule_dim'])/r

            output_dim = num_capsules*capsule_dim
            primary_capsules = tf.layers.dense(
                encoded,
                output_dim,
                use_bias=False
            )
            primary_capsules = tf.reshape(
                primary_capsules,
                [encoded.shape[0].value,
                 tf.shape(encoded)[1],
                 num_capsules,
                 capsule_dim]
            )

            primary_capsules = ops.squash(primary_capsules)
            prim_norm = ops.safe_norm(primary_capsules)

            tf.add_to_collection('image', tf.expand_dims(prim_norm, 3))
            primary_capsules = tf.identity(primary_capsules, 'primary_capsules')

        return primary_capsules

    def tc_capsules(self, primary_capsules, seq_length):
        '''
        get the time coded capsules

        args:
            - primary_capsules:
                [batch_size x time x num_capsules x capsule_dim]

        returns:
            - the time coded capsules:
                [batch_size x time' x num_capsules x capsule_dim]
        '''

        with tf.variable_scope('time_coded_capsules'):

            capsules = tf.identity(primary_capsules, 'primary_capsules')

            num_capsules = primary_capsules.shape[2].value
            capsule_dim = primary_capsules.shape[3].value
            width = int(self.conf['width'])
            stride = int(self.conf['stride'])

            for l in range(int(self.conf['num_tc_layers'])):
                with tf.variable_scope('layer%d' % l):

                    num_capsules /= int(self.conf['capsule_ratio'])
                    capsule_dim *= int(self.conf['capsule_ratio'])

                    capsules = layers.conv1d_capsule(
                        inputs=capsules,
                        num_capsules=num_capsules,
                        capsule_dim=capsule_dim,
                        width=width,
                        stride=stride,
                        routing_iters=int(self.conf['routing_iters'])
                    )
                    seq_length -= width - 1
                    seq_length /= stride

                    norm = ops.safe_norm(capsules)
                    tf.add_to_collection(
                        'image', tf.expand_dims(norm, 3, 'tc_norm%d' % l))

            capsules = tf.identity(capsules, 'time_coded_capsules')

        return capsules, seq_length

    def rc_capsules(self, tc_capsules):
        '''get the output capsules

        args:
            tc_capsules: time coded capsules
                [batch_size x time x num_capsules x capsule_dim]


        returns:
            - the rated coded capsules
                [batch_size x num_capsules x capsule_dim]
            - the contribution of each timestep in the rate coded capsules
                [batch_size x time x num_capsules x capsules_dim]
        '''

        with tf.variable_scope('rate_coded_capsules'):

            capsules = tf.identity(tc_capsules, 'tc_capsules')

            r = int(self.conf['capsule_ratio'])**int(self.conf['num_rc_layers'])
            num_capsules = self.num_output_capsules*r
            capsule_dim = int(self.conf['output_capsule_dim'])/r

            layer = layers.TCRCCapsule(
                num_capsules=num_capsules,
                capsule_dim=capsule_dim,
                routing_iters=int(self.conf['routing_iters'])
            )

            capsules = layer(capsules)

            #get the predictions
            predictions = tf.get_default_graph().get_tensor_by_name(
                layer.scope_name + '/predict/transpose_1:0')

            #get the final squash factor
            sf = tf.get_default_graph().get_tensor_by_name(
                layer.scope_name + '/cluster/squash/div_1:0')

            #get the final routing weights
            logits = tf.get_default_graph().get_tensor_by_name(
                layer.scope_name + '/cluster/while/Exit_1:0')
            weights = layer.probability_fn(logits)
            weights *= tf.transpose(sf, [0, 2, 1])
            input_capsules = tc_capsules.shape[2].value
            weights = tf.stack(tf.split(weights, input_capsules, 1), 2)

            #compute the contributions
            contrib = tf.reduce_sum(predictions*tf.expand_dims(weights, 4), 2)

            contrib = tf.identity(contrib, 'contrib')
            capsules = tf.identity(capsules, 'output_capsules')

        return capsules, contrib

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

            capsules = tf.identity(rc_capsules, 'rc_capsules')
            contrib = tf.identity(contrib, 'contrib')
            num_capsules = capsules.shape[1].value
            capsule_dim = capsules.shape[2].value

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
                    contrib_predict = layer.predict(contrib)

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
            all_caps = tf.concat(tf.unstack(output_capsules, axis=1), 1)

            for i, val in enumerate(valids):
                with tf.variable_scope(val):
                    alignment = alignments[:, :, i]
                    alignment = tf.expand_dims(alignment, 2)
                    p = tf.layers.dense(
                        all_caps,
                        len(valids[val]),
                        tf.nn.sigmoid,
                        name=val)
                    alignment *= tf.expand_dims(tf.square(p), 1)
                    p *= ops.safe_norm(output_capsules[:, i], keepdims=True)
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
