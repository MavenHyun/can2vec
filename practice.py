import tensorflow as tf
# NumPy is often used to load, manipulate and preprocess data.
import numpy as np
import pandas as pd
import statistics as st
import math

class practice_autoencoder:

    def __init__(self, h1, h2, h3, h4, h5, h6, lr1, lr2, lr3):
        self.num_hnodes_mut1 = h1
        self.num_hnodes_mut2 = h2
        self.num_hnodes_CNV1 = h3
        self.num_hnodes_CNV2 = h4
        self.num_hnodes_mRNA1 = h5
        self.num_hnodes_mRNA2 = h6
        self.rate_learning1 = lr1
        self.rate_learning2 = lr2
        self.rate_learning3 = lr3

    def data_preprocess(self, file_name, file_name2):
        df = pd.read_csv(file_name, '\t')
        df.fillna(0, inplace=True)
        self.X_categ, self.X_mut, self.X_CNV, self.X_mRNA = np.array(df.values[:19, 1:]).transpose(), np.array(df.values[19:128, 1:]).transpose(), np.array(df.values[128:2348, 1:]).transpose(), np.array(df.values[2348:18872, 1:]).transpose()

        mean, stdv = st.mean(self.X_categ[:, 0]), st.stdev(self.X_categ[:, 0])
        for i in range(self.X_categ.shape[0]):
            self.X_categ[i, 0] = (self.X_categ[i, 0] - mean) / stdv

        for i in range(self.X_mRNA.shape[1]):
            mean, stdv = st.mean(self.X_mRNA[:, i]), st.stdev(self.X_mRNA[:, i])
            if (stdv != 0):
                for j in range(self.X_mRNA.shape[0]):
                    self.X_mRNA[j, i] = (self.X_mRNA[j, i] - mean) / stdv

        '''
        df = pd.read_csv(file_name2, '\t')
        raw_input = df.values[:, 1:92]
        self.Y = np.array(raw_input).transpose()
        mean, stdv = st.mean(self.Y[:, 0]), st.stdev(self.Y[:, 0])

        for i in range(self.num_samples):
            self.Y[i, 0] = (self.Y[i, 0] - mean) / stdv
        '''

    def data_training(self):

        #input_categ = tf.placeholder("float", [self.X_categ.shape[0], self.X_categ.shape[1]])
        input_mut = tf.placeholder("float", [self.X_mut.shape[0], self.X_mut.shape[1]])
        input_CNV = tf.placeholder("float", [self.X_CNV.shape[0], self.X_CNV.shape[1]])
        input_mRNA = tf.placeholder("float", [self.X_mRNA.shape[0], self.X_mRNA.shape[1]])

        weights = {
            'W_input_mut_hidden1': tf.Variable(tf.random_normal([self.X_mut.shape[1], self.num_hnodes_mut1])),
            'W_hidden1_mut_hidden2': tf.Variable(tf.random_normal([self.num_hnodes_mut1, self.num_hnodes_mut2])),
            'W_hidden2_mut_output': tf.Variable(tf.random_normal([self.num_hnodes_mut2, self.X_mut.shape[1]])),
            'W_input_CNV_hidden1': tf.Variable(tf.random_normal([self.X_CNV.shape[1], self.num_hnodes_CNV1])),
            'W_hidden1_CNV_hidden2': tf.Variable(tf.random_normal([self.num_hnodes_CNV1, self.num_hnodes_CNV2])),
            'W_hidden2_CNV_output': tf.Variable(tf.random_normal([self.num_hnodes_CNV2, self.X_CNV.shape[1]])),
            'W_input_mRNA_hidden1': tf.Variable(tf.random_normal([self.X_mRNA.shape[1], self.num_hnodes_mRNA1])),
            'W_hidden1_mRNA_hidden2': tf.Variable(tf.random_normal([self.num_hnodes_mRNA1, self.num_hnodes_mRNA2])),
            'W_hidden2_mRNA_output': tf.Variable(tf.random_normal([self.num_hnodes_mRNA2, self.X_mRNA.shape[1]]))
        }

        bias = {
            'B_input_mut_hidden1' : tf.Variable(tf.random_normal([self.num_hnodes_mut1])),
            'B_hidden1_mut_hidden2' : tf.Variable(tf.random_normal([self.num_hnodes_mut2])),
            'B_hidden2_mut_output' : tf.Variable(tf.random_normal([self.X_mut.shape[1]])),
            'B_input_CNV_hidden1': tf.Variable(tf.random_normal([self.num_hnodes_CNV1])),
            'B_hidden1_CNV_hidden2': tf.Variable(tf.random_normal([self.num_hnodes_CNV2])),
            'B_hidden2_CNV_output': tf.Variable(tf.random_normal([self.X_CNV.shape[1]])),
            'B_input_mRNA_hidden1': tf.Variable(tf.random_normal([self.num_hnodes_mRNA1])),
            'B_hidden1_mRNA_hidden2': tf.Variable(tf.random_normal([self.num_hnodes_mRNA2])),
            'B_hidden2_mRNA_output': tf.Variable(tf.random_normal([self.X_mRNA.shape[1]]))
        }

        hidden1_mut = tf.nn.sigmoid(tf.add(tf.matmul(input_mut, weights['W_input_mut_hidden1']), bias['B_input_mut_hidden1']))
        hidden2_mut = tf.nn.sigmoid(tf.add(tf.matmul(hidden1_mut, weights['W_hidden1_mut_hidden2']), bias['B_hidden1_mut_hidden2']))
        output_mut = tf.nn.sigmoid(tf.add(tf.matmul(hidden2_mut, weights['W_hidden2_mut_output']), bias['B_hidden2_mut_output']))

        hidden1_CNV = tf.nn.sigmoid(tf.add(tf.matmul(input_CNV, weights['W_input_CNV_hidden1']), bias['B_input_CNV_hidden1']))
        hidden2_CNV = tf.nn.sigmoid(tf.add(tf.matmul(hidden1_CNV, weights['W_hidden1_CNV_hidden2']), bias['B_hidden1_CNV_hidden2']))
        output_CNV = tf.nn.sigmoid(tf.add(tf.matmul(hidden2_CNV, weights['W_hidden2_CNV_output']), bias['B_hidden2_CNV_output']))
        
        hidden1_mRNA = tf.nn.sigmoid(tf.add(tf.matmul(input_mRNA, weights['W_input_mRNA_hidden1']), bias['B_input_mRNA_hidden1']))
        hidden2_mRNA = tf.nn.sigmoid(tf.add(tf.matmul(hidden1_mRNA, weights['W_hidden1_mRNA_hidden2']), bias['B_hidden1_mRNA_hidden2']))
        output_mRNA = tf.nn.sigmoid(tf.add(tf.matmul(hidden2_mRNA, weights['W_hidden2_mRNA_output']), bias['B_hidden2_mRNA_output']))

        cost_mut = tf.reduce_mean(tf.pow(input_mut - output_mut, 2))
        optimizer_mut = tf.train.GradientDescentOptimizer(self.rate_learning1).minimize(cost_mut)
        cost_CNV = tf.reduce_mean(tf.pow(input_CNV - output_CNV, 2))
        optimizer_CNV = tf.train.GradientDescentOptimizer(self.rate_learning1).minimize(cost_CNV)
        cost_mRNA = tf.reduce_mean(tf.pow(input_mRNA - output_mRNA, 2))
        optimizer_mRNA = tf.train.GradientDescentOptimizer(self.rate_learning1).minimize(cost_mRNA)

        init = tf.global_variables_initializer()
        sess = tf.Session()
        sess.run(init)

        """
        for epoch in range(400):
            for i in range(256):
                batch_xs, batch_ys = mnist.train.next_batch(256)
                _, cost_value = sess.run([optimizer, cost], feed_dict={input: batch_xs})
            if epoch % 1 == 0:
                print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(cost_value))
        print("Stacked Autoencoder pre-training Optimization Finished!")
        """

        for iter in range(10000):
           cost1, cost2, cost3, _, _, _ = sess.run([cost_mut, cost_CNV, cost_mRNA, optimizer_mut, optimizer_CNV, optimizer_mRNA],
                                    feed_dict={input_mut: self.X_mut, input_CNV: self.X_CNV, input_mRNA: self.X_mRNA})
           if (iter % 100 == 0):
               print(iter,  "\tMutation Data\t", cost1, "\tCNV Data\t", cost2, "\tmRNA Data\t", cost3)

test = practice_autoencoder(20, 20, 75, 75, 300, 300, 0.80, 0.80, 0.80)
test.data_preprocess('ACC_features.tsv', 'ACC_survival.tsv')
test.data_training()


