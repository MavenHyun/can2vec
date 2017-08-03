import tensorflow as tf
import numpy as np
import os

def snp_noise(dataset, fraction, switch):

    denoised = dataset.copy()
    if switch is True:
        for i in range(dataset.shape[0]):
            mask = np.random.randint(0, dataset.shape[1], fraction)
            for m in mask:
                denoised[i, m] = 0.
    return denoised

class split_ae:

    def __init__(self, data_set, dim_hid, learn_rate, denoise):
        with tf.name_scope("Input_Layer"):
            self.X_train, self.X_eval, self.X_test = data_set[0], data_set[1], data_set[2]
            self.H_dim, self.L_rate, self.F_num, self.D_mode = dim_hid, learn_rate, data_set[0].shape[1], denoise
            self.input_data = tf.placeholder("float", [None, self.X_train.shape[1]])

        with tf.name_scope("Encoding_Layer"):
            self.W_enc = tf.Variable(tf.random_normal([self.F_num, self.H_dim]))
            self.B_enc = tf.Variable(tf.random_normal([self.H_dim]))
            self.enc = tf.nn.relu(tf.add(tf.matmul(self.input_data, self.W_enc), self.B_enc))

        with tf.name_scope("Decoding_Layer"):
            self.W_dec = tf.Variable(tf.random_normal([self.H_dim, self.F_num]))
            self.B_dec = tf.Variable(tf.random_normal([self.F_num]))
            self.dec = tf.nn.relu(tf.add(tf.matmul(self.enc, self.W_dec), self.B_dec))
            self.answer_data = tf.placeholder("float", [None, self.F_num])

        with tf.name_scope("Optimus_Prime"):
            self.C = tf.reduce_mean(tf.pow(self.input_data - self.dec, 2))
            self.O = tf.train.AdamOptimizer(self.L_rate).minimize(self.C)

    def initiate(self):
        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init)
            diff, num_train, delta = 1.0, 0, 100.0
            while num_train < 1000:
                num_train += 1
                train_cost, _, self.W, self.B = sess.run([self.C, self.O, self.W_enc, self.B_enc],
                                                         feed_dict={self.input_data:
                                                                    snp_noise(self.X_train, 0.3, self.D_mode),
                                                                    self.answer_data: self.X_train})
                eval_cost = sess.run(self.C, feed_dict={self.input_data: self.X_eval, self.answer_data: self.X_eval})
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
            test_cost = sess.run(self.C, feed_dict={self.input_data: self.X_test, self.answer_data: self.X_test})
        self.result_iter = num_train
        self.result_test = test_cost

    def printout(self):
        print("iter: ", self.result_iter, " result: ", self.result_test)