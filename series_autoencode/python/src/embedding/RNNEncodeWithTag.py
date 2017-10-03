# -*- coding: utf-8 -*-

import tensorflow as tf


class RNNEncodeWithTag(object):
    """
    Encoding with tags
    """
    def __init__(self, data, data_length,
                 target, hidden_num,
                 layer_num, max_series,
                 learning_rate, activation):

        class_num = int(target.get_shape()[1])

        # 隐藏层
        cell = tf.contrib.rnn.MultiRNNCell(
          [tf.contrib.rnn.BasicLSTMCell(hidden_num, activation=activation) for _ in range(layer_num)]
        )
        encode_output, _ = tf.nn.dynamic_rnn(cell, data, sequence_length=data_length, dtype=tf.float32)

        # 只要RNN最后一个输出
        self.last_encode_output = tf.gather(encode_output, axis=1, indices=max_series-1)

        # 输出层
        weight = tf.Variable(tf.truncated_normal([hidden_num, class_num]))
        bias = tf.Variable(tf.constant(0.1, shape=[class_num]))

        # 定义损失
        self.prediction = tf.matmul(self.last_encode_output, weight) + bias
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.prediction, labels=target))
        print("Prediction", self.prediction)
        print("Target", target)
        print("Loss", self.loss)
        self.softmax_prediction = tf.nn.softmax(self.prediction)
        self.final_prediction = tf.argmax(self.softmax_prediction, 1)

        # 选择优化器
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
        self.train = optimizer.minimize(self.loss)

        # 定义错误
        mistakes = tf.not_equal(tf.argmax(target, 1), tf.argmax(self.prediction, 1))
        self.error = tf.reduce_mean(tf.cast(mistakes, tf.float32))
