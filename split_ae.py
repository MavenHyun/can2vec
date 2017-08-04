import tensorflow as tf
import numpy as np
import tf_components as tc
import os

class split_ae:

    def __init__(self, data_set, denoise):
        with tf.name_scope("Basic_Settings"):
            self.X_train, self.X_eval, self.X_test = data_set[0], data_set[1], data_set[2]
            self.F_num, self.D_mode, self.D_min = data_set[0].shape[1], denoise, 0

        with tf.name_scope("Placeholders"):
            self.input_data = tf.placeholder("float", [None, self.F_num])
            self.answer_data = tf.placeholder("float", [None, self.F_num])

        with tf.name_scope("Dictionaries"):
            self.train_dict = {self.input_data: self.X_train, self.answer_data: self.X_train}
            self.eval_dict = {self.input_data: self.X_eval, self.answer_data: self.X_eval}
            self.test_dict = {self.input_data: self.X_test, self.answer_data: self.X_test}

    def construct_Encoder(self, layer_name, input_ph, dim0, dim1, activation):
        enc = tc.Encoder(layer_name, input_ph, dim0, dim1, activation)
        return enc

    def construct_Decoder(self, layer_name, input_ph, dim0, dim1, activation):
        dec = tc.Decoder(layer_name, input_ph, dim0, dim1, activation)
        return dec

    def construct_Optimizer(self, layer_name, output_ph, answer_ph, learn_rate, train_meth):
        opt = tc.Optimizer(layer_name, output_ph, answer_ph, learn_rate, train_meth)
        return opt

    def initiate(self, optimizer, force_epochs, encoder):
        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init)
            products = optimizer.optimize_sa(sess, self.train_dict, self.eval_dict, self.test_dict, force_epochs, encoder)
        return products

    def print_result(self, optimizer):
        print("Number of Epochs: ", optimizer.result_iter, " Test Result: ", optimizer.result_test)

