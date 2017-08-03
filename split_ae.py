import tensorflow as tf
import numpy as np
import tf_components as tc
import os

class split_ae:

    def __init__(self, data_set, dim_hid, learn_rate, denoise):
        with tf.name_scope("Input_Layer"):
            self.X_train, self.X_eval, self.X_test = data_set[0], data_set[1], data_set[2]
            self.H_dim, self.L_rate, self.F_num, self.D_mode = dim_hid, learn_rate, data_set[0].shape[1], denoise
            self.input_data = tf.placeholder("float", [None, self.F_num])
            self.answer_data = tf.placeholder("float", [None, self.F_num])
        '''
        with tf.name_scope("Encoding_Layer"):
            self.W_enc = tf.Variable(tf.random_normal([self.F_num, self.H_dim]))
            self.B_enc = tf.Variable(tf.random_normal([self.H_dim]))
            self.enc = tf.nn.relu(tf.add(tf.matmul(self.input_data, self.W_enc), self.B_enc))

        with tf.name_scope("Decoding_Layer"):
            self.W_dec = tf.Variable(tf.random_normal([self.H_dim, self.F_num]))
            self.B_dec = tf.Variable(tf.random_normal([self.F_num]))
            self.dec = tf.nn.relu(tf.add(tf.matmul(self.enc, self.W_dec), self.B_dec))

        with tf.name_scope("Optimus_Prime"):
            self.answer_data = tf.placeholder("float", [None, self.F_num])
            self.C = tf.reduce_mean(tf.pow(self.answer_data - self.dec, 2))
            self.O = tf.train.AdamOptimizer(self.L_rate).minimize(self.C)
        '''
    def construct_encoder(self, layer_name, input_ph, dim0, dim1, activation):
        enc = tc.encoder_ae(layer_name, input_ph, dim0, dim1, activation)
        return enc

    def construct_decoder(self, layer_name, input_ph, dim0, dim1, activation):
        dec = tc.encoder_ae(layer_name, input_ph, dim0, dim1, activation)
        return dec

    def construct_optimizer(self, opti_name, output_ph, answer_ph, learn_rate, train_meth):
        opt = tc.optimizer_ae(opti_name, output_ph, answer_ph, learn_rate, train_meth)
        return opt

    def initiate(self, get_W, get_B, opt):
        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init)
            opt.train_getWB #organize placeholders and dicts

    def printout(self):
        print("iter: ", self.result_iter, " result: ", self.result_test)