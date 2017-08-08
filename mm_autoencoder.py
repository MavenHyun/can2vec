import tensorflow as tf
import numpy as np
import train_set as ts

class model:

    def __init__(self, dataset, learn):
        self.N = dataset.X.shape[0]
        self.F_cli = dataset.X_cli.shape[1]
        self.F_mut = dataset.X_mut.shape[1]
        self.F_CNV = dataset.X_CNV.shape[1]
        self.F_mRNA = dataset.X_mRNA.shape[1]

        with tf.name_scope("Input_Layer"):
            self.x = { 'cli': tf.placeholder("float", [92, None]),
                       'mut': tf.placeholder("float", [92, None]),
                       'CNV': tf.placeholder("float", [92, None]),
                       'mRNA': tf.placeholder("float", [92, None]) }


        with tf.name_scope("Weights_and_Bias"):
            self.w = { 'mut': tf.Variable(tf.random_normal([self.F_mut, 41])),
                       'CNV': tf.Variable(tf.random_normal([self.F_CNV, 814])),
                       'mRNA': tf.Variable(tf.random_normal([self.F_mRNA, 8014])) }

            self.b = { 'mut': tf.Variable(tf.random_normal([41])),
                       'CNV': tf.Variable(tf.random_normal([814])),
                       'mRNA': tf.Variable(tf.random_normal([8014])) }

    def encode(self, feature):
        with tf.name_scope(feature + "_Encoded"):
            result = tf.nn.relu(tf.add(tf.matmul(self.x[feature], self.w[feature]), self.b[feature]))
        return result

    def concat(self):
        with tf.name_scope("Feature_Vector"):
            result = tf.concat(self.encode())







