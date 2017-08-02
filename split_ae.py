import tensorflow as tf
import os

class split_ae:

    def __init__(self, x, y, z, dh, lr):
        with tf.name_scope("Input_Layer"):
            self.X, self.Y, self.Z, self.dim_h, self.learning_rate = x, y, z, dh, lr
            self.input_data = tf.placeholder("float", [None, self.X.shape[1]])

        with tf.name_scope("Encoding_Layer"):
            self.W_enc = tf.Variable(tf.random_normal([self.X.shape[1], self.dim_h]))
            self.B_enc = tf.Variable(tf.random_normal([self.dim_h]))
            self.enc = tf.nn.relu(tf.add(tf.matmul(self.input_data, self.W_enc), self.B_enc))

        with tf.name_scope("Decoding_Layer"):
            self.W_dec = tf.Variable(tf.random_normal([self.dim_h, self.X.shape[1]]))
            self.B_dec = tf.Variable(tf.random_normal([self.X.shape[1]]))
            self.dec = tf.nn.relu(tf.add(tf.matmul(self.enc, self.W_dec), self.B_dec))

        with tf.name_scope("Optimus_Prime"):
            self.C = tf.reduce_mean(tf.pow(self.input_data - self.dec, 2))
            self.O = tf.train.AdamOptimizer(self.learning_rate).minimize(self.C)

    def initiate(self):
        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init)
            diff, num_train, delta = 1.0, 0, 100.0
            while diff > 0.01 and num_train < 10000 and delta > 0.00001:
                num_train += 1
                train_cost, _, self.W, self.B = sess.run([self.C, self.O, self.W_enc, self.B_enc],
                                                feed_dict={self.input_data: self.X})
                eval_cost = sess.run(self.C, feed_dict={self.input_data: self.Y})
                diff = abs(eval_cost - train_cost)
                if num_train % 500 == 0:
                    print(num_train, "th step:\t", "Cost is ", train_cost)
                    print("Evaluation: ", eval_cost)
                    print(diff)
                if num_train == 1:
                    old_diff = diff
                else:
                    delta = abs(old_diff - diff)
                    old_diff = diff
            test_cost = sess.run(self.C, feed_dict={self.input_data: self.Z})
            print("Number of iterations: ", num_train, " Test: ", test_cost)
        self.result_iter = num_train
        self.result_diff = diff
        self.result_test = test_cost

    def printout(self):
        print(self.result_iter)
        print(self.result_diff)
        print(self.result_test)