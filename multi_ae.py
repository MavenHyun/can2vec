import tensorflow as tf
# NumPy is often used to load, manipulate and preprocess data.
import numpy as np
import pandas as pd
import statistics as st


class train_set:
    def __init__(self, fn, fn2):
        self.file_name, self.file_name2 = fn, fn2

    def data_preprocess(self):
        df = pd.read_csv(self.file_name, '\t')
        df.fillna(0, inplace=True)

        self.X, self.X_categ, self.X_mut, self.X_CNV, self.X_mRNA = np.array(df.values[:18872, 1:]).transpose(), \
                                                                    np.array(df.values[:19, 1:]).transpose(), \
                                                                    np.array(df.values[19:128, 1:]).transpose(), \
                                                                    np.array(df.values[128:2348, 1:]).transpose(), \
                                                                    np.array(df.values[2348:18872, 1:]).transpose()
        mean, stdv = st.mean(self.X_categ[:, 0]), st.stdev(self.X_categ[:, 0])
        for i in range(self.X_categ.shape[0]):
            self.X_categ[i, 0] = (self.X_categ[i, 0] - mean) / stdv
        for i in range(self.X_mRNA.shape[1]):
            mean, stdv = st.mean(self.X_mRNA[:, i]), st.stdev(self.X_mRNA[:, i])
            if (stdv != 0):
                for j in range(self.X_mRNA.shape[0]):
                    self.X_mRNA[j, i] = (self.X_mRNA[j, i] - mean) / stdv

        df = pd.read_csv(self.file_name2, '\t')
        raw_input = df.values[:, 1:]
        self.Y = np.array(raw_input).transpose()
        mean, stdv = st.mean(self.Y[:, 0]), st.stdev(self.Y[:, 0])
        for i in range(self.Y.shape[0]):
            self.Y[i, 0] = (self.Y[i, 0] - mean) / stdv

        self.Z = np.concatenate((self.X, self.Y), 1)


class multi_ae:
    def __init__(self, set, h1, h2, h3, h4, lr):
        self.rate_learning = lr
        self.X, self.Y, self.Z = set.X, set.Y, set.Z
        self.X_categ, self.X_mut, self.X_CNV, self.X_mRNA = set.X_categ, set.X_mut, set.X_CNV, set.X_mRNA
        self.h_categ, self.h_mut, self.h_CNV, self.h_mRNA = h1, h2, h3, h4
        self.h = h1 + h2 + h3 + h4

    def tf_construct(self, set):
        self.input_categ = tf.placeholder("float", [set.X_categ.shape[0], set.X_categ.shape[1]])
        self.input_mut = tf.placeholder("float", [set.X_mut.shape[0], set.X_mut.shape[1]])
        self.input_CNV = tf.placeholder("float", [set.X_CNV.shape[0], set.X_CNV.shape[1]])
        self.input_mRNA = tf.placeholder("float", [set.X_mRNA.shape[0], set.X_mRNA.shape[1]])
        self.input_surviv = tf.placeholder("float", [set.Y.shape[0], set.Y.shape[1]])
        self.weights = {
            'W_categ_enc': tf.Variable(tf.random_normal([self.X_categ.shape[1], self.h_categ])),
            'W_mut_enc': tf.Variable(tf.random_normal([self.X_mut.shape[1], self.h_mut])),
            'W_CNV_enc': tf.Variable(tf.random_normal([self.X_CNV.shape[1], self.h_CNV])),
            'W_mRNA_enc': tf.Variable(tf.random_normal([self.X_mRNA.shape[1], self.h_mRNA])),
            'W_enc': tf.Variable(tf.random_normal([self.X.shape[1], self.h])),
            'W_categ_dec': tf.Variable(tf.random_normal([self.h_categ, self.X_categ.shape[1]])),
            'W_mut_dec': tf.Variable(tf.random_normal([self.h_mut, self.X_mut.shape[1]])),
            'W_CNV_dec': tf.Variable(tf.random_normal([self.h_CNV, self.X_CNV.shape[1]])),
            'W_mRNA_dec': tf.Variable(tf.random_normal([self.h_mRNA, self.X_mRNA.shape[1]])),
            'W_dec': tf.Variable(tf.random_normal([self.h, self.Z.shape[1]]))
        }
        self.bias = {
            'B_categ_enc': tf.Variable(tf.random_normal([self.h_categ])),
            'B_mut_enc': tf.Variable(tf.random_normal([self.h_mut])),
            'B_CNV_enc': tf.Variable(tf.random_normal([self.h_CNV])),
            'B_mRNA_enc': tf.Variable(tf.random_normal([self.h_mRNA])),
            'B_enc': tf.Variable(tf.random_normal([self.h])),
            'B_categ_dec': tf.Variable(tf.random_normal([self.X_categ.shape[1]])),
            'B_mut_dec': tf.Variable(tf.random_normal([self.X_mut.shape[1]])),
            'B_CNV_dec': tf.Variable(tf.random_normal([self.X_CNV.shape[1]])),
            'B_mRNA_dec': tf.Variable(tf.random_normal([self.X_mRNA.shape[1]])),
            'B_dec': tf.Variable(tf.random_normal([self.Z.shape[1]]))
        }
        self.hiddens = {
            'H_categ': tf.nn.sigmoid(
                tf.add(tf.matmul(self.input_categ, self.weights['W_categ_enc']), self.bias['B_categ_enc'])),
            'H_mut': tf.nn.sigmoid(
                tf.add(tf.matmul(self.input_mut, self.weights['W_mut_enc']), self.bias['B_mut_enc'])),
            'H_CNV': tf.nn.sigmoid(
                tf.add(tf.matmul(self.input_CNV, self.weights['W_CNV_enc']), self.bias['B_CNV_enc'])),
            'H_mRNA': tf.nn.sigmoid(
                tf.add(tf.matmul(self.input_mRNA, self.weights['W_mRNA_enc']), self.bias['B_mRNA_enc']))
        }
        self.outputs = {
            'output_categ': tf.nn.sigmoid(
                tf.add(tf.matmul(self.hiddens['H_categ'], self.weights['W_categ_dec']), self.bias['B_categ_dec'])),
            'output_mut': tf.nn.sigmoid(
                tf.add(tf.matmul(self.hiddens['H_mut'], self.weights['W_mut_dec']), self.bias['B_mut_dec'])),
            'output_CNV': tf.nn.sigmoid(
                tf.add(tf.matmul(self.hiddens['H_CNV'], self.weights['W_CNV_dec']), self.bias['B_CNV_dec'])),
            'output_mRNA': tf.nn.sigmoid(
                tf.add(tf.matmul(self.hiddens['H_mRNA'], self.weights['W_mRNA_dec']), self.bias['B_mRNA_dec']))
        }
        self.costs = {
            'cost_categ': tf.reduce_mean(tf.pow(self.input_categ - self.outputs['output_categ'], 2)),
            'cost_mut': tf.reduce_mean(tf.pow(self.input_mut - self.outputs['output_mut'], 2)),
            'cost_CNV': tf.reduce_mean(tf.pow(self.input_CNV - self.outputs['output_CNV'], 2)),
            'cost_mRNA': tf.reduce_mean(tf.pow(self.input_mRNA - self.outputs['output_mRNA'], 2))
        }
        self.optis = {
            'opti_categ': tf.train.RMSPropOptimizer(self.rate_learning).minimize(self.costs['cost_categ']),
            'opti_mut': tf.train.RMSPropOptimizer(self.rate_learning).minimize(self.costs['cost_mut']),
            'opti_CNV': tf.train.RMSPropOptimizer(self.rate_learning).minimize(self.costs['cost_CNV']),
            'opti_mRNA': tf.train.RMSPropOptimizer(self.rate_learning).minimize(self.costs['cost_mRNA'])
        }

        feature = tf.concat(
            [self.hiddens['H_categ'], self.hiddens['H_mut'], self.hiddens['H_CNV'], self.hiddens['H_mRNA']], 1)
        output = tf.nn.sigmoid(tf.add(tf.matmul(feature, self.weights['W_dec']), self.bias['B_dec']))
        output0, output1 = tf.split(output, [self.Z.shape[1] - 1, 1], 1)
        self.cost = tf.reduce_mean(tf.pow(self.input_surviv - output1, 2))
        self.opt = tf.train.RMSPropOptimizer(self.rate_learning).minimize(self.cost)

    def pre_train(self):
        init = tf.global_variables_initializer()
        sess = tf.Session()
        sess.run(init)
        for iter in range(1000):
            _, _, _, _, _, _, _, _ = sess.run([self.costs['cost_categ'], self.optis['opti_categ'],
                                               self.costs['cost_mut'], self.optis['opti_mut'],
                                               self.costs['cost_CNV'], self.optis['opti_CNV'],
                                               self.costs['cost_mRNA'], self.optis['opti_mRNA']],
                                              feed_dict={self.input_categ: self.X_categ,
                                                         self.input_mut: self.X_mut,
                                                         self.input_CNV: self.X_CNV,
                                                         self.input_mRNA: self.X_mRNA})
        print("Optimization done!")

    def data_train(self):
        init = tf.global_variables_initializer()
        sess = tf.Session()
        sess.run(init)

        for iter in range(5001):
            c, _ = sess.run([self.cost, self.opt], feed_dict={self.input_categ: self.X_categ,
                                                              self.input_mut: self.X_mut,
                                                              self.input_CNV: self.X_CNV,
                                                              self.input_mRNA: self.X_mRNA,
                                                              self.input_surviv: self.Y})
            if iter % 100 == 0:
                print(iter, "Cost is ", c)

    def execute(self, set):
        self.tf_construct(set)
        self.pre_train()
        self.data_train()