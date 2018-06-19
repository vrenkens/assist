'''@file encoder_decoder.py
Contains the EncoderDecoder class'''

import tfmodel
import tensorflow as tf

class EncoderDecoder(tfmodel.TFModel):
    '''an encoder-decoder with dynamic routing acquisition model'''


    def model(self, inputs, seq_length):
        '''apply the model'''

        #encode the features
        encoded, seq_length = self.encoder(inputs, seq_length)

        #decode the encoded features
        probs = self.decoder(encoded, seq_length)

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

    def decoder(self, encoded, seq_length):
        '''decode the encoded features

        args:
            encoded: a [N x T x F] tensor
            seq_length: encoded sequence length

        returns:
            - the label probabilities [B x L]
        '''

        with tf.variable_scope('decoder'):

            mask = tf.sequence_mask(seq_length, tf.shape(encoded)[1])
            mask = tf.tile(
                mask[:, :, tf.newaxis],
                [1, 1, encoded.shape[-1].value])
            encoded = tf.where(mask, encoded,
                               tf.ones_like(encoded)*encoded.dtype.min)
            outputs = tf.reduce_max(encoded, 1)
            outputs = tf.layers.dense(
                outputs,
                int(self.conf['numunits_decoder']),
                tf.nn.relu)
            outputs = tf.layers.dense(outputs, self.coder.numlabels,
                                      tf.nn.sigmoid)
            return outputs
