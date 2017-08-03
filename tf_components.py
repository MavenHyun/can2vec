import tensorflow as tf
import numpy as np

class encoder_ae:
    def __init__(self, layer_name, input_ph, dim0, dim1, activation):
        with tf.name_scope(layer_name):
            self.weights = tf.Variable(tf.random_normal([dim0, dim1]))
            self.bias = tf.Variable(tf.random_normal([dim1]))
            if activation == 0:
                self.result = tf.nn.relu(tf.add(tf.matmul(input_ph, self.weights), self.bias))
            elif activation == 1:
                self.result = tf.nn.sigmoid(tf.add(tf.matmul(input_ph, self.weights), self.bias))
            elif activation == 2:
                self.result = tf.nn.tanh(tf.add(tf.matmul(input_ph, self.weights), self.bias))
            else:
                self.result = tf.add(tf.matmul(input_ph, self.weights), self.bias)

class decoder_ae:
    def __init__(self, layer_name, input_ph, dim0, dim1, activation):
        with tf.name_scope(layer_name):
            self.weights = tf.Variable(tf.random_normal([dim0, dim1]))
            self.bias = tf.Variable(tf.random_normal([dim1]))
            if activation == 0:
                self.result = tf.nn.relu(tf.add(tf.matmul(input_ph, self.weights), self.bias))
            elif activation == 1:
                self.result = tf.nn.sigmoid(tf.add(tf.matmul(input_ph, self.weights), self.bias))
            elif activation == 2:
                self.result = tf.nn.tanh(tf.add(tf.matmul(input_ph, self.weights), self.bias))
            else:
                self.result = tf.add(tf.matmul(input_ph, self.weights), self.bias)

class optimizer_ae:
    def __init__(self, opti_name, output_ph, answer_ph, learn_rate, train_meth):
        with tf.name_scope(opti_name):
            self.cost = tf.reduce_mean(tf.pow(answer_ph - output_ph, 2))
            if train_meth == 0:
                self.opti = tf.train.AdamOptimizer(learn_rate).minimize(self.cost)
            elif train_meth == 1:
                self.opti = tf.train.RMSPropOptimizer(learn_rate).minimize(self.cost)
            elif train_meth == 2:
                self.opti = tf.train.AdagradOptimizer(learn_rate).minimize(self.cost)
            elif train_meth == 3:
                self.opti = tf.train.AdadeltaOptimizer(learn_rate).minimize(self.cost)
            else:
                self.opti = tf.train.GradientDescentOptimizer(learn_rate).minimize(self.cost)