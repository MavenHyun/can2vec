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
        self.weights, self.bias, self.result_iter, self.result_test = 0, 0, 0, 0
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

    def optimize_test(self, session, train_dict, eval_dict, test_dict, force_epochs):
        diff, num_train, delta = 1.0, 0, 100.0
        while num_train < force_epochs:
            num_train += 1
            train_cost, _, = session.run([self.cost, self.opti], feed_dict=train_dict)
            eval_cost = session.run(self.cost, feed_dict=eval_dict)
            diff = abs(eval_cost - train_cost)
            if num_train == 1:
                old_diff = diff
            else:
                delta, old_diff = abs(old_diff - diff), diff
            if num_train == 1000:
                if num_train < 10000 or delta > 0.00001 or diff > 0.001:
                    break
                else:
                    num_train -= 1
        test_cost = session.run(self.cost, feed_dict=test_dict)
        self.result_iter = num_train
        self.result_test = test_cost

    def get_optimized(self, session, weights, bias, test_dict):
        opt_weights, opt_bias = session.run([weights, bias], feed_dict=test_dict)
        opt_parameters = [opt_weights, opt_bias]
        return opt_parameters