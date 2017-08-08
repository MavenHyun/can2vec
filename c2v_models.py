import tensorflow as tf
import numpy as np
import train_set as ts


def black_magic(operation, name):
    black_spell = {
        'relu': tf.nn.relu(operation),
        'sigmoid': tf.nn.sigmoid(operation),
        'tanh': tf.nn.tanh(operation)
    }
    return black_spell[name]

def white_magic(name, learn, cost):
    white_spell = {
        'adam': tf.train.AdamOptimizer(learn).minimize(cost),
        'rms': tf.train.RMSPropOptimizer(learn).minimize(cost),
        'adag': tf.train.AdagradOptimizer(learn).minimize(cost),
        'adad': tf.train.AdadeltaOptimizer(learn).minimize(cost),
        'grad': tf.train.GradientDescentOptimizer(learn).minimize(cost)
    }
    return white_spell[name]
    
class NoviceSeer:
    #vali = 50, test = 30
    def __init__(self, dataset, num1, num2):
        
        with tf.name_scope("Basic_Settings"):
            self.D, self.N, self.S = dataset.X, dataset.X['all'].shape[0], dataset.Y
            self.P, self.W, self.B, self.R = {}, {}, {}, {}
            self.train_dict, self.vali_dict, self.test_dict = {}, {}, {}
            self.size_train = num1
            self.size_test = num1 + num2
            
    def leading_encoder(self, fea, dim, fun):
        
        with tf.name_scope(fea + "_Encoder"):
            self.P[fea] = tf.placeholder("float", [None, None])
            self.d = np.split(self.D[fea], [self.size_train, self.size_test], axis=0)
            self.train_dict[self.P[fea]] = np.split(self.D[fea], [self.size_train, self.size_test, self.N], axis=0)[0]
            self.vali_dict[self.P[fea]] = np.split(self.D[fea], [self.size_train, self.size_test, self.N], axis=0)[1]
            self.test_dict[self.P[fea]] = np.split(self.D[fea], [self.size_train, self.size_test, self.N], axis=0)[2]

            self.W[fea] = tf.Variable(tf.random_normal([self.D[fea].shape[1], dim]))
            self.B[fea] = tf.Variable(tf.random_normal([dim]))
            self.R[fea] = black_magic(tf.add(tf.matmul(self.P[fea], self.W[fea]), self.B[fea]), fun)
                                                    
            return self.R[fea]

    def surv_predictor(self, target, dim, fun):
        
        with tf.name_scope("Survivability_Predictor"):
            self.P['surviv'] = tf.placeholder("float", [None, 1])
            self.train_dict[self.P['surviv']] = np.split(self.S, [self.size_train, self.size_test, self.N], axis=0)[0]
            self.vali_dict[self.P['surviv']] = np.split(self.S, [self.size_train, self.size_test, self.N], axis=0)[1]
            self.test_dict[self.P['surviv']] = np.split(self.S, [self.size_train, self.size_test, self.N], axis=0)[2]

            self.W['surviv'] = tf.Variable(tf.random_normal([dim, 1]))
            self.B['surviv'] = tf.Variable(tf.random_normal([1]))
            self.R['surviv'] = black_magic(tf.add(tf.matmul(target, self.W['surviv']), self.B['surviv']), fun)

            return self.R['surviv']

    def foresight(self, result, answer, meth, epochs, learn):

        cost = tf.reduce_mean(tf.pow(result - answer, 2))
        opti = white_magic(meth, learn, cost)

        for iter in range(epochs):
            init = tf.global_variables_initializer()
            with tf.Session() as sess:
                sess.run(init)
                train_cost, _ = sess.run([cost, opti], feed_dict=self.train_dict)
                vali_cost = sess.run(cost, feed_dict=self.vali_dict)
                if iter % 100 == 0:
                    print("Training Cost: ", train_cost, "Evaluation Cost: ", vali_cost)
        test_cost = sess.run(cost, feed_dict=self.test_dict)
        print("Test Cost: ",  test_cost)



            

