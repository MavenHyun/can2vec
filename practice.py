import tensorflow as tf
# NumPy is often used to load, manipulate and preprocess data.
import numpy as np
import pandas as pd
import statistics as st

def mask_noise(dataset, fraction):

    denoised = dataset.copy()
    for i in range(dataset.shape[0]):
        mask = np.random.randint(0, dataset.shape[1], fraction)
        for m in mask:
            denoised[i, m] = 0.
    return denoised

def snp_noise(dataset, fraction):

    denoised = dataset.copy()
    min = dataset.min()
    max = dataset.max()
    for i, sample in enumerate(dataset):
        mask = np.random.randint(0, dataset.shape[1], fraction)
        for m in mask:
            if np.random.random() < 0.5:
                denoised[i, m] = min
            else:
                denoised[i, m] = max
    return denoised

class basic_ae:

    def __init__(self, h, lr, dn):
        self.num_hnodes = h
        self.rate_learning = lr
        self.denoise = dn

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

        self.Z = np.concatenate((self.X_categ, self.X_mut, self.X_CNV, self.X_mRNA, self.Y), 1)
        if (self.denoise == True):
            self.X_categ = mask_noise(self.X_categ, 5)
            self.X_mut = mask_noise(self.X_mut, 50)
            self.X_CNV = mask_noise(self.X_CNV, 1000)
            self.X_mRNA = mask_noise(self.X_mRNA, 3000)
            self.Y = mask_noise(self.Y, 45)

    def data_training(self, file_name, file_name2):
        self.data_preprocess(file_name, file_name2)

        input_categ = tf.placeholder("float", [self.X_categ.shape[0], self.X_categ.shape[1]])
        input_mut = tf.placeholder("float", [self.X_mut.shape[0], self.X_mut.shape[1]])
        input_CNV = tf.placeholder("float", [self.X_CNV.shape[0], self.X_CNV.shape[1]])
        input_mRNA = tf.placeholder("float", [self.X_mRNA.shape[0], self.X_mRNA.shape[1]])
        input_surviv = tf.placeholder("float", [self.Y.shape[0], self.Y.shape[1]])
        input_all = tf.concat([input_categ, input_mut, input_CNV, input_mRNA, input_surviv], 1)
        answer = tf.placeholder("float", [self.X.shape[0], self.X.shape[1] + self.Y.shape[1]])

        weights = {
            'W_enc': tf.Variable(tf.random_normal([self.X.shape[1] + self.Y.shape[1], self.num_hnodes])),
            'W_dec': tf.Variable(tf.random_normal([self.num_hnodes, self.X.shape[1] + self.Y.shape[1]]))
        }
        bias = {
            'B_enc': tf.Variable(tf.random_normal([self.num_hnodes])),
            'B_dec': tf.Variable(tf.random_normal([self.X.shape[1] + self.Y.shape[1]]))
        }

        hidden = tf.nn.sigmoid(tf.add(tf.matmul(input_all, weights['W_enc']), bias['B_enc']))
        output = tf.nn.sigmoid(tf.add(tf.matmul(hidden, weights['W_dec']), bias['B_dec']))
        cost = tf.reduce_mean(tf.pow(answer - output, 2))
        optimizer = tf.train.RMSPropOptimizer(self.rate_learning).minimize(cost)

        init = tf.global_variables_initializer()
        sess = tf.Session()
        sess.run(init)

        for iter in range(5000):
            cost_value, _ = sess.run(
                [cost, optimizer],
                feed_dict={input_categ: self.X_categ,
                           input_mut: self.X_mut,
                           input_CNV: self.X_CNV,
                           input_mRNA: self.X_mRNA,
                           input_surviv: self.Y,
                           answer: self.Z})
            if (iter % 100 == 0):
                print(iter, "\tCost value is ", cost_value)

class stacked_ae:

    def __init__(self, h1, h2, lr, dn):
        self.num_hnodes1 = h1
        self.num_hnodes2 = h2
        self.rate_learning = lr
        self.denoise = dn

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

        self.Z = np.concatenate((self.X_categ, self.X_mut, self.X_CNV, self.X_mRNA, self.Y), 1)
        if (self.denoise == True):
            self.X_categ = mask_noise(self.X_categ, 5)
            self.X_mut = mask_noise(self.X_mut, 50)
            self.X_CNV = mask_noise(self.X_CNV, 1000)
            self.X_mRNA = mask_noise(self.X_mRNA, 3000)
            self.Y = mask_noise(self.Y, 45)

    def data_training(self, file_name, file_name2):
        self.data_preprocess(file_name, file_name2)
        input_categ = tf.placeholder("float", [self.X_categ.shape[0], self.X_categ.shape[1]])
        input_mut = tf.placeholder("float", [self.X_mut.shape[0], self.X_mut.shape[1]])
        input_CNV = tf.placeholder("float", [self.X_CNV.shape[0], self.X_CNV.shape[1]])
        input_mRNA = tf.placeholder("float", [self.X_mRNA.shape[0], self.X_mRNA.shape[1]])
        input_surviv = tf.placeholder("float", [self.Y.shape[0], self.Y.shape[1]])
        input_all = tf.concat([input_categ, input_mut, input_CNV, input_mRNA, input_surviv], 1)
        answer = tf.placeholder("float", [self.X.shape[0], self.Z.shape[1]])

        weights = {
            'W_enc1': tf.Variable(tf.random_normal([self.X.shape[1] + self.Y.shape[1], self.num_hnodes1])),
            'W_enc2': tf.Variable(tf.random_normal([self.num_hnodes1, self.num_hnodes2])),
            'W_dec2': tf.Variable(tf.random_normal([self.num_hnodes2, self.num_hnodes1])),
            'W_dec1': tf.Variable(tf.random_normal([self.num_hnodes1, self.Z.shape[1]]))
        }
        bias = {
            'B_enc1': tf.Variable(tf.random_normal([self.num_hnodes1])),
            'B_enc2': tf.Variable(tf.random_normal([self.num_hnodes2])),
            'B_dec2': tf.Variable(tf.random_normal([self.num_hnodes1])),
            'B_dec1': tf.Variable(tf.random_normal([self.Z.shape[1]]))
        }

        hidden1 = tf.nn.sigmoid(tf.add(tf.matmul(input_all, weights['W_enc1']), bias['B_enc1']))
        hidden2 = tf.nn.sigmoid(tf.add(tf.matmul(hidden1, weights['W_enc2']), bias['B_enc2']))
        hidden3 = tf.nn.sigmoid(tf.add(tf.matmul(hidden2, weights['W_dec2']), bias['B_dec2']))
        output = tf.nn.sigmoid(tf.add(tf.matmul(hidden3, weights['W_dec1']), bias['B_dec1']))
        cost = tf.reduce_mean(tf.pow(answer - output, 2))
        optimizer = tf.train.RMSPropOptimizer(self.rate_learning).minimize(cost)

        init = tf.global_variables_initializer()
        sess = tf.Session()
        sess.run(init)

        for iter in range(5000):
            cost_value, _ = sess.run(
                [cost, optimizer],
                feed_dict={input_categ: self.X_categ,
                           input_mut: self.X_mut,
                           input_CNV: self.X_CNV,
                           input_mRNA: self.X_mRNA,
                           input_surviv: self.Y,
                           answer: self.Z})
            if (iter % 100 == 0):
                print(iter, "\tCost value is ", cost_value)








class multi_ae:

    def __init__(self, h1, h2, h3, lr, dn):
        self.num_hnodes_mut = h1
        self.num_hnodes_CNV = h2
        self.num_hnodes_mRNA = h3
        self.rate_learning = lr
        self.denoise = dn

    def data_preprocess(self, file_name, file_name2):
        df = pd.read_csv(file_name, '\t')
        df.fillna(0, inplace=True)
        self.X_categ, self.X_mut, self.X_CNV, self.X_mRNA = np.array(df.values[:19, 1:]).transpose(), \
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

        self.Z = self.Y
        if (self.denoise == True):
            self.X_categ = mask_noise(self.X_categ, 5)
            self.X_mut = mask_noise(self.X_mut, 50)
            self.X_CNV = mask_noise(self.X_CNV, 1000)
            self.X_mRNA = mask_noise(self.X_mRNA, 3000)
            self.Y = mask_noise(self.Y, 45)
            self.num_hnodes = self.num_hnodes_mut + self.num_hnodes_CNV + self.num_hnodes_mRNA \
                          + self.X_categ.shape[1] + self.Y.shape[1]

    def data_training(self, file_name, file_name2):

        self.data_preprocess(file_name, file_name2)
        input_categ = tf.placeholder("float", [self.X_categ.shape[0], self.X_categ.shape[1]])
        input_mut = tf.placeholder("float", [self.X_mut.shape[0], self.X_mut.shape[1]])
        input_CNV = tf.placeholder("float", [self.X_CNV.shape[0], self.X_CNV.shape[1]])
        input_mRNA = tf.placeholder("float", [self.X_mRNA.shape[0], self.X_mRNA.shape[1]])
        input_surviv = tf.placeholder("float", [self.Y.shape[0], self.Y.shape[1]])
        answer = tf.placeholder("float", [self.Y.shape[0], self.Y.shape[1]])

        weights = {
            'W_mut_enc': tf.Variable(tf.random_normal([self.X_mut.shape[1], self.num_hnodes_mut])),
            'W_CNV_enc': tf.Variable(tf.random_normal([self.X_CNV.shape[1], self.num_hnodes_CNV])),
            'W_mRNA_enc': tf.Variable(tf.random_normal([self.X_mRNA.shape[1], self.num_hnodes_mRNA])),
            'W_mut_dec': tf.Variable(tf.random_normal([self.num_hnodes_mut, self.X_mut.shape[1]])),
            'W_CNV_dec': tf.Variable(tf.random_normal([self.num_hnodes_CNV, self.X_CNV.shape[1]])),
            'W_mRNA_dec': tf.Variable(tf.random_normal([self.num_hnodes_mRNA, self.X_mRNA.shape[1]])),
            'W_dec': tf.Variable(tf.random_normal([self.num_hnodes, self.Z.shape[1]]))
        }

        bias = {
            'B_mut_enc': tf.Variable(tf.random_normal([self.num_hnodes_mut])),
            'B_CNV_enc': tf.Variable(tf.random_normal([self.num_hnodes_CNV])),
            'B_mRNA_enc': tf.Variable(tf.random_normal([self.num_hnodes_mRNA])),
            'B_mut_dec': tf.Variable(tf.random_normal([self.X_mut.shape[1]])),
            'B_CNV_dec': tf.Variable(tf.random_normal([self.X_CNV.shape[1]])),
            'B_mRNA_dec': tf.Variable(tf.random_normal([self.X_mRNA.shape[1]])),
            'B_dec': tf.Variable(tf.random_normal([self.Z.shape[1]]))
        }

        hidden_mut = tf.nn.sigmoid(tf.add(tf.matmul(input_mut, weights['W_mut_enc']), bias['B_mut_enc']))
        output_mut = tf.nn.sigmoid(tf.add(tf.matmul(hidden_mut, weights['W_mut_dec']), bias['B_mut_dec']))
        hidden_CNV = tf.nn.sigmoid(tf.add(tf.matmul(input_CNV, weights['W_CNV_enc']), bias['B_CNV_enc']))
        output_CNV = tf.nn.sigmoid(tf.add(tf.matmul(hidden_CNV, weights['W_CNV_dec']), bias['B_CNV_dec']))
        hidden_mRNA = tf.nn.sigmoid(tf.add(tf.matmul(input_mRNA, weights['W_mRNA_enc']), bias['B_mRNA_enc']))
        output_mRNA = tf.nn.sigmoid(tf.add(tf.matmul(hidden_mRNA, weights['W_mRNA_dec']), bias['B_mRNA_dec']))
        feature = tf.concat([input_categ, hidden_mut, hidden_CNV, hidden_mRNA, input_surviv], 1)
        hidden = tf.nn.sigmoid(tf.add(tf.matmul(feature, weights['W_dec']), bias['B_dec']))
        hidden0, output = tf.split(hidden, [self.Z.shape[1] - 1, 1], 1)
        '''
        cost_mut = tf.reduce_mean(tf.pow(input_mut - output_mut, 2))
        optimizer_mut = tf.train.RMSPropOptimizer(self.rate_learning).minimize(cost_mut)
        cost_CNV = tf.reduce_mean(tf.pow(input_CNV - output_CNV, 2))
        optimizer_CNV = tf.train.RMSPropOptimizer(self.rate_learning).minimize(cost_CNV)
        cost_mRNA = tf.reduce_mean(tf.pow(input_mRNA - output_mRNA, 2))
        optimizer_mRNA = tf.train.RMSPropOptimizer(self.rate_learning).minimize(cost_mRNA)
        '''


        cost_surviv = tf.reduce_mean(tf.pow(answer - output, 2))
        optimizer_surviv = tf.train.RMSPropOptimizer(self.rate_learning).minimize(cost_surviv)

        init = tf.global_variables_initializer()
        sess = tf.Session()
        sess.run(init)

        for iter in range(5000):
           #cost1, cost2, cost3, cost4, _, _, _, _ = sess.run([cost_mut, cost_CNV, cost_mRNA, cost_surviv, optimizer_mut, optimizer_CNV, optimizer_mRNA,  optimizer_surviv],
           cost_value, _ = sess.run([cost_surviv, optimizer_surviv],
                                    feed_dict={input_categ: self.X_categ, input_mut: self.X_mut,
                                               input_CNV: self.X_CNV, input_mRNA: self.X_mRNA,
                                               input_surviv: self.Y, answer: self.Z})
           if (iter % 100 == 0):
               #print(iter,  "\tMutation Data\t", cost1, "\tCNV Data\t", cost2, "\tmRNA Data\t", cost3, "\tSurvivability\t", cost4)
               print(iter, "cost is ", cost_value)

test = multi_ae(20, 100, 500, 0.80, True)
test.data_training('ACC_features.tsv', 'ACC_survival.tsv')



