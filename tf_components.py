import tensorflow as tf
import numpy as np

class Encoder:
    def __init__(self, layer_name, input_ph, dim0, dim1, activation):
        with tf.name_scope(layer_name):
            self.shape0, self.shape1 = dim0, dim1
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

class Decoder:
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

class Projector:
    def __init__(self, layer_name, input_ph, dim, activation):
        with tf.name_scope(layer_name):
            self.weights = tf.Variable(tf.random_normal([dim, dim]))
            self.bias = tf.Variable(tf.random_normal([dim]))
            if activation == 0:
                self.result = tf.nn.relu(tf.add(tf.matmul(input_ph, self.weights), self.bias))
            elif activation == 1:
                self.result = tf.nn.sigmoid(tf.add(tf.matmul(input_ph, self.weights), self.bias))
            elif activation == 2:
                self.result = tf.nn.tanh(tf.add(tf.matmul(input_ph, self.weights), self.bias))
            else:
                self.result = tf.add(tf.matmul(input_ph, self.weights), self.bias)

class Predictor:
    def __init__(self, layer_name, input_ph, dim, activation):
        with tf.name_scope(layer_name):
            self.weights = tf.Variable(tf.random_normal([dim, 1]))
            self.bias = tf.Variable(tf.random_normal([1]))
            if activation == 0:
                self.result = tf.nn.relu(tf.add(tf.matmul(input_ph, self.weights), self.bias))
            elif activation == 1:
                self.result = tf.nn.sigmoid(tf.add(tf.matmul(input_ph, self.weights), self.bias))
            elif activation == 2:
                self.result = tf.nn.tanh(tf.add(tf.matmul(input_ph, self.weights), self.bias))
            else:
                self.result = tf.add(tf.matmul(input_ph, self.weights), self.bias)

class ReConstructor:
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

class Optimizer:
    def __init__(self, layer_name, output_ph, answer_ph, learn_rate, train_meth):
        self.weights, self.bias, self.result_iter, self.result_test, self.products = 0, 0, 0, 0, 0
        with tf.name_scope(layer_name):
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

    def optimize_sa(self, session, train_dict, eval_dict, test_dict, epochs, encoder):
        diff, num_train, delta = 1.0, 0, 100.0
        while num_train < epochs:
            num_train += 1
            train_cost, _, = session.run([self.cost, self.opti], feed_dict=train_dict)
            eval_cost = session.run(self.cost, feed_dict=eval_dict)
            diff = abs(eval_cost - train_cost)
            if num_train == 1:
                old_diff = diff
            else:
                delta, old_diff = abs(old_diff - diff), diff
            if num_train > 500 and (delta > 0.00001 or diff > 0.00001):
                break
            if num_train % 10 == 0:
                print("Training Cost:", train_cost, "Evaluation Cost: ", eval_cost)
        test_cost = session.run(self.cost, feed_dict=test_dict)
        self.result_iter = num_train
        self.result_test = test_cost
        product_train = session.run(encoder.result, feed_dict=train_dict)
        product_eval = session.run(encoder.result, feed_dict=eval_dict)
        product_test = session.run(encoder.result, feed_dict=test_dict)
        return [product_train, product_eval, product_test]

    def optimize_ma(self, session, train_dict, eval_dict, test_dict, epochs):
        for iter in range(epochs):
            train_cost, _, = session.run([self.cost, self.opti], feed_dict=train_dict)
            eval_cost = session.run(self.cost, feed_dict=eval_dict)
            if iter % 100 == 0:
                print("Training Cost: ", train_cost, "Evaluation Cost: ", eval_cost)
        test_cost = session.run(self.cost, feed_dict=test_dict)
        print("Test Cost: ", test_cost)
