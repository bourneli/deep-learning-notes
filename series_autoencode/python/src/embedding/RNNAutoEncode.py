# -*- coding: utf-8 -*-

import tensorflow as tf


class RNNAutoEncode(object):
    """
    Reference paper https://arxiv.org/abs/1409.3215
    """
    def __init__(self, series, series_length,
                 hidden_num, feature_num, max_series,
                 learning_rate=0.0001, layer_num=1,
                 activation=tf.nn.relu):

        print("Encode input Shape", series.get_shape())

        # Encoding layer
        encode_cell = tf.contrib.rnn.MultiRNNCell(
          [tf.contrib.rnn.BasicLSTMCell(hidden_num, reuse=False, activation=activation)
           for _ in range(layer_num)]
        )
        encode_output, self.encode_final_state = tf.nn.dynamic_rnn(
          encode_cell, series, sequence_length=series_length, dtype=tf.float32, scope='encode')

        encode_weight = tf.Variable(tf.truncated_normal([hidden_num, feature_num]))
        encode_bias = tf.Variable(tf.constant(0.1, shape=[feature_num]))
        last_encode_output = tf.gather(encode_output, axis=1, indices=max_series-1)
        last_encode_output = tf.expand_dims(last_encode_output, axis=1)
        last_encode_output = tf.map_fn(lambda out: tf.matmul(out, encode_weight)+encode_bias, last_encode_output)
        print("Last encode output shape", last_encode_output.get_shape())

        # Decoding layer
        # remove the first unit feature of each input data
        # input_without_first = tf.slice(series, begin=[0, 1, 0],
        #                                size=[-1, max_series - 1, feature_num])
        input_without_last = tf.slice(series,
                                      begin=[0, 0, 0],
                                      size=[-1, max_series - 1, feature_num])
        decode_input = tf.concat([last_encode_output, input_without_last], axis=1)
        print("Decode input  shape", decode_input.get_shape())

        decode_cell = tf.contrib.rnn.MultiRNNCell(
          [tf.contrib.rnn.BasicLSTMCell(hidden_num, reuse=False, activation=activation)
           for _ in range(layer_num)]
        )
        decode_cell_output, _ = tf.nn.dynamic_rnn(decode_cell, decode_input,
                                                  sequence_length=series_length,
                                                  initial_state=self.encode_final_state,
                                                  dtype=tf.float32, scope='decode')
        print("Decode cell output shape", decode_cell_output.get_shape())

        decode_weight = tf.Variable(tf.truncated_normal([hidden_num, feature_num]))
        decode_bias = tf.Variable(tf.constant(0.1, shape=[feature_num]))
        padding_weight = tf.map_fn(
          lambda length: tf.concat(
            [tf.ones(shape=(length, feature_num), dtype=tf.float32),
             tf.zeros(shape=(max_series - length, feature_num), dtype=tf.float32)], axis=0),
          series_length,
          dtype=tf.float32)
        print("Padding Weight", padding_weight)
        decode_output = padding_weight*tf.map_fn(lambda unit: tf.matmul(unit, decode_weight)+decode_bias,
                                                 decode_cell_output)
        print("Decode output shape", decode_output.get_shape())

        # Loss Function
        self.loss = tf.losses.mean_squared_error(labels=series, predictions=decode_output)
        print("Loss", self.loss)

        # Optimizer
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
        self.train = optimizer.minimize(self.loss)
