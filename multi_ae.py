import tensorflow as tf
# NumPy is often used to load, manipulate and preprocess data.
import numpy as np
import pandas as pd
import statistics as st

class split_ae:
    def __init__(self, a, h, lr):
        self.X = a
        self.num_hnodes = h
        self.rate_learning = lr
        
    def pretrain(self):
        x = tf.placeholder("float", [self.X.shape[0], self.X.shape[1]])
        self.weights = {
            'W_enc': tf.Variable(tf.random_normal([self.X.shape[1], self.num_hnodes])),
            'W_dec': tf.Variable(tf.random_normal([self.num_hnodes, self.X.shape[1]]))
        }
        self.bias = {
            'B_enc': tf.Variable(tf.random_normal([self.num_hnodes])),
            'B_dec': tf.Variable(tf.random_normal([self.X.shape[1]]))
        }
        self.hidden = tf.nn.sigmoid(tf.add(tf.matmul(x, self.weights['W_enc']), self.bias['B_enc']))
        y = tf.nn.sigmoid(tf.add(tf.matmul(self.hidden, self.weights['W_dec']), self.bias['B_dec']))
        cost = tf.reduce_mean(
            tf.pow(x - y, 2))
        opt = tf.train.RMSPropOptimizer(self.rate_learning).minimize(cost)
        init = tf.global_variables_initializer()
        sess = tf.Session()
        sess.run(init)
        for iter in range(1001):
            _, _ = sess.run(
                [cost, opt],
                feed_dict={x: self.X})
        print("Optimization done!")

    def get_H(self):
        return self.hidden
    

class multi_ae:
    def __init__(self, h1, h2, h3, lr):
        self.num_hnodes_mut = h1
        self.num_hnodes_CNV = h2
        self.num_hnodes_mRNA = h3
        self.rate_learning = lr

    def data_preprocess(self, file_name, file_name2):
        df = pd.read_csv(file_name, '\t')
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

        df = pd.read_csv(file_name2, '\t')
        raw_input = df.values[:, 1:]
        self.Y = np.array(raw_input).transpose()
        mean, stdv = st.mean(self.Y[:, 0]), st.stdev(self.Y[:, 0])
        for i in range(self.Y.shape[0]):
            self.Y[i, 0] = (self.Y[i, 0] - mean) / stdv
        self.num_hnodes = self.num_hnodes_mut + self.num_hnodes_CNV + self.num_hnodes_mRNA + 10
        self.Z = np.concatenate((self.X, self.Y), 1)

    def data_training(self, file_name, file_name2):
        self.data_preprocess(file_name, file_name2)
        input_categ = tf.placeholder("float", [self.X_categ.shape[0], self.X_categ.shape[1]])
        input_mut = tf.placeholder("float", [self.X_mut.shape[0], self.X_mut.shape[1]])
        input_CNV = tf.placeholder("float", [self.X_CNV.shape[0], self.X_CNV.shape[1]])
        input_mRNA = tf.placeholder("float", [self.X_mRNA.shape[0], self.X_mRNA.shape[1]])
        input_surviv = tf.placeholder("float", [self.Y.shape[0], self.Y.shape[1]])
        answer = tf.placeholder("float", [self.Z.shape[0], self.Z.shape[1]])

        split_categ = split_ae(self.X_categ, 10, 0.80)
        split_mut = split_ae(self.X_mut, self.num_hnodes_mut, 0.80)
        split_CNV = split_ae(self.X_CNV, self.num_hnodes_CNV, 0.80)
        split_mRNA = split_ae(self.X_mRNA, self.num_hnodes_mRNA, 0.80)
        split_categ.pretrain(), split_mut.pretrain(), split_CNV.pretrain(), split_mRNA.pretrain()

        W_dec = tf.Variable(tf.random_normal([self.num_hnodes, self.Z.shape[1]]))
        B_dec = tf.Variable(tf.random_normal([self.Z.shape[1]]))
        feature = tf.concat([split_categ.get_H(), split_mut.get_H(), split_CNV.get_H(), split_mRNA.get_H()], 1)
        output = tf.nn.sigmoid(tf.add(tf.matmul(feature, W_dec), B_dec))

        cost = tf.reduce_mean(tf.pow(answer - output, 2))
        opt = tf.train.RMSPropOptimizer(self.rate_learning).minimize(cost)
        init = tf.global_variables_initializer()
        sess = tf.Session()
        sess.run(init)

        for iter in range(5001):
            cost, _ = sess.run([cost, opt],
                               feed_dict={input_categ: self.X_categ, input_mut: self.X_mut,
                                          input_CNV: self.X_CNV, input_mRNA: self.X_mRNA,
                                          input_surviv: self.Y, answer: self.Z})
            if iter % 100 == 0:
                print(iter, "Cost is ", cost)