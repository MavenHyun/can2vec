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

def grey_magic(learn, old_train, new_train):
    if 0 <= old_train - new_train < 0.001:
        mod_learn = learn * 0.3
    else:
        mod_learn = learn
    return mod_learn

def red_magic(learn, old_train, new_train, old_vali, new_vali, iter):
    if abs(new_train - new_vali) < 0.1 and new_train < 1 and new_vali < 1 and iter > 20000:
        return True
    elif 0 <= old_vali - new_vali < 0.000001 and  0 <= old_train - new_train < 0.000001 and iter > 20000:
        return True
    elif learn < 1.0e-30:
        return True
    else:
        return False

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

            self.W[fea] = tf.Variable(tf.truncated_normal([self.D[fea].shape[1], dim]))
            self.B[fea] = tf.Variable(tf.truncated_normal([dim]))
            result = black_magic(tf.add(tf.matmul(self.P[fea], self.W[fea]), self.B[fea]), fun)
                                                    
            return result

    def data_projector(self, target, dim, fun):

        with tf.name_scope("Data_Projector"):
            self.W['proj'] = tf.Variable(tf.truncated_normal([dim, dim]))
            self.B['proj'] = tf.Variable(tf.truncated_normal([dim]))
            self.R['proj'] = black_magic(tf.add(tf.matmul(target, self.W['proj']), self.B['proj']), fun)

            return self.R['proj']

    def surv_predictor(self, target, dim, fun):
        
        with tf.name_scope("Survivability_Predictor"):
            self.P['surviv'] = tf.placeholder("float", [None, 1])
            self.train_dict[self.P['surviv']] = np.split(self.S, [self.size_train, self.size_test, self.N], axis=0)[0]
            self.vali_dict[self.P['surviv']] = np.split(self.S, [self.size_train, self.size_test, self.N], axis=0)[1]
            self.test_dict[self.P['surviv']] = np.split(self.S, [self.size_train, self.size_test, self.N], axis=0)[2]

            self.W['surviv'] = tf.Variable(tf.truncated_normal([dim, 1]))
            self.B['surviv'] = tf.Variable(tf.truncated_normal([1]))
            self.R['surviv'] = black_magic(tf.add(tf.matmul(target, self.W['surviv']), self.B['surviv']), fun)

            return self.R['surviv']

    def foresight(self, result, answer, meth, epochs, learn):

        cost = tf.reduce_mean(tf.pow(result - answer, 2))
        opti = white_magic(meth, learn, cost)

        with tf.Session() as sess:
            train_writer = tf.summary.FileWriter("C:/Users/Arthur Keonwoo Kim/PycharmProjects/can2vec/output/",
                                                 sess.graph)
            init = tf.global_variables_initializer()
            sess.run(init)
            for iter in range(epochs):
                train_cost, _ = sess.run([cost, opti], feed_dict=self.train_dict)
                vali_cost = sess.run(cost, feed_dict=self.vali_dict)
                if iter % 500 == 0:
                    print("Training Cost: ", train_cost, "Evaluation Cost: ", vali_cost)
            test_cost = sess.run(cost, feed_dict=self.test_dict)
            print("Test Cost: ",  test_cost)

class SeerAdept:
    # vali = 50, test = 30
    def __init__(self, dataset, num1, num2):
        with tf.name_scope("Basic_Settings"):
            self.D, self.N, self.S = dataset.X, dataset.X['all'].shape[0], dataset.Y
            self.P, self.W, self.B, self.R = {}, {}, {}, {}
            self.train_dict, self.vali_dict, self.test_dict = {}, {}, {}
            self.size_train = num1
            self.size_test = num1 + num2

    def training_encoder(self, fea, dim, fun):
        with tf.name_scope(fea + "_EncT"):
            self.P[fea] = tf.placeholder("float", [None, None])
            self.train_dict[self.P[fea]] = np.split(self.D[fea], [self.size_train, self.size_test, self.N], axis=0)[0]
            self.vali_dict[self.P[fea]] = np.split(self.D[fea], [self.size_train, self.size_test, self.N], axis=0)[1]
            self.test_dict[self.P[fea]] = np.split(self.D[fea], [self.size_train, self.size_test, self.N], axis=0)[2]
            self.W[fea + '_encT'] = tf.get_variable(fea + "_encT_W", shape=[self.D[fea].shape[1], dim],
                                                    initializer=tf.contrib.layers.xavier_initializer())
            self.B[fea + '_encT'] = tf.get_variable(fea + "_encT_B", shape=[dim],
                                                    initializer=tf.contrib.layers.xavier_initializer())
            result = black_magic(tf.add(tf.matmul(self.P[fea], self.W[fea + '_encT']), self.B[fea + '_encT']), fun)
        return result

    def training_decoder(self, enc, fea, dim, fun):
        with tf.name_scope(fea + "_DecT"):
            self.W[fea + '_decT'] = tf.get_variable(fea + "_decT_W", shape=[dim, self.D[fea].shape[1]],
                                                    initializer=tf.contrib.layers.xavier_initializer())
            self.B[fea + '_decT'] = tf.get_variable(fea + "_decT_B", shape=[self.D[fea].shape[1]],
                                                    initializer=tf.contrib.layers.xavier_initializer())
            result = black_magic(tf.add(tf.matmul(enc, self.W[fea + '_decT']), self.B[fea + '_decT']), fun)
        return result

    def training_autoencoder(self, fea, dim, fun):
        with tf.name_scope(fea + "_AutoT"):
            enc = self.training_encoder(fea, dim, fun)
            dec = self.training_decoder(enc, fea, dim, fun)
        return dec

    def leading_encoder(self, fea, fun):
        with tf.name_scope(fea + "_Encoder"):
            self.P[fea] = tf.placeholder("float", [None, None])
            self.train_dict[self.P[fea]] = np.split(self.D[fea], [self.size_train, self.size_test, self.N], axis=0)[0]
            self.vali_dict[self.P[fea]] = np.split(self.D[fea], [self.size_train, self.size_test, self.N], axis=0)[1]
            self.test_dict[self.P[fea]] = np.split(self.D[fea], [self.size_train, self.size_test, self.N], axis=0)[2]

            self.W[fea + '_encL'] = self.W[fea + '_encT']
            self.B[fea + '_encL'] = self.B[fea + '_encT']
            self.R[fea + '_encL'] = black_magic(tf.add(tf.matmul(self.P[fea], self.W[fea + '_encL']),
                                                       self.B[fea + '_encL']), fun)

            return self.R[fea + '_encL']

    def data_projector1(self, target, dim0, dim1, fun):
        with tf.name_scope("Data_Projector1"):
            self.W['proj1'] = tf.Variable(tf.truncated_normal([dim0, dim1]))
            self.B['proj1'] = tf.Variable(tf.truncated_normal([dim1]))
            self.R['proj1'] = black_magic(tf.add(tf.matmul(target, self.W['proj1']), self.B['proj1']), fun)
        #Individualization is needed!
            return self.R['proj1']

    def data_projector2(self, target, dim0, dim1, fun):
        with tf.name_scope("Data_Projector2"):
            self.W['proj2'] = tf.Variable(tf.truncated_normal([dim0, dim1]))
            self.B['proj2'] = tf.Variable(tf.truncated_normal([dim1]))
            self.R['proj2'] = black_magic(tf.add(tf.matmul(target, self.W['proj2']), self.B['proj2']), fun)
        #Individualization is needed!
            return self.R['proj2']

    def surv_predictor(self, target, dim, fun):
        with tf.name_scope("Survivability_Predictor"):
            self.P['surviv'] = tf.placeholder("float", [None, 1])
            self.train_dict[self.P['surviv']] = np.split(self.S, [self.size_train, self.size_test, self.N], axis=0)[0]
            self.vali_dict[self.P['surviv']] = np.split(self.S, [self.size_train, self.size_test, self.N], axis=0)[1]
            self.test_dict[self.P['surviv']] = np.split(self.S, [self.size_train, self.size_test, self.N], axis=0)[2]

            self.W['surviv'] = tf.get_variable("surviv_W", shape=[dim, 1],
                                               initializer=tf.contrib.layers.xavier_initializer())
            self.B['surviv'] = tf.get_variable("surviv_B", shape=[1],
                                               initializer=tf.contrib.layers.xavier_initializer())
            self.R['surviv'] = black_magic(tf.add(tf.matmul(target, self.W['surviv']), self.B['surviv']), fun)

            return self.R['surviv']

    def mirror_image(self, result, answer, meth, epochs, learn):
        with tf.name_scope("Encoder_Optimizer"):
            cost = tf.reduce_mean(tf.pow(result - answer, 2))
            opti = white_magic(meth, learn, cost)
            old_train, old_vali = 0, 0
            with tf.Session() as sess:
                train_writer = tf.summary.FileWriter("C:/Users/Arthur Keonwoo Kim/PycharmProjects/can2vec/output/",
                                                     sess.graph)
                init = tf.global_variables_initializer()
                sess.run(init)
                for iter in range(epochs):
                    train_cost, _ = sess.run([cost, opti], feed_dict=self.train_dict)
                    vali_cost = sess.run(cost, feed_dict=self.vali_dict)
                    if iter % 500 == 0:
                        learn = grey_magic(learn, train_cost, old_train)
                        print(iter, "Training Cost: ", train_cost, "Evaluation Cost: ", vali_cost)
                        old_train = train_cost
                    if red_magic(learn, train_cost, old_vali, vali_cost, iter) is True:
                        break
                test_cost = sess.run(cost, feed_dict=self.test_dict)
                print("Test Cost: ", test_cost, "Training Cost: ", train_cost, "Evaluation Cost: ", vali_cost,
                      "Final Learning Rate: ", learn)

    def foresight(self, result, answer, meth, epochs, learn):
        with tf.name_scope("Survivability_Optimizer"):
            cost = tf.reduce_mean(tf.pow(result - answer, 2))
            opti = white_magic(meth, learn, cost)
            old_train, old_vali = 0, 0
            with tf.Session() as sess:
                train_writer = tf.summary.FileWriter("C:/Users/Arthur Keonwoo Kim/PycharmProjects/can2vec/output/",
                                                     sess.graph)
                init = tf.global_variables_initializer()
                sess.run(init)
                for iter in range(epochs):
                    train_cost, _ = sess.run([cost, opti], feed_dict=self.train_dict)
                    vali_cost = sess.run(cost, feed_dict=self.vali_dict)
                    if iter % 500 == 0:
                        learn = grey_magic(learn, train_cost, old_train)
                        print(iter, "Training Cost: ", train_cost, "Evaluation Cost: ", vali_cost)
                        old_train = train_cost
                    if red_magic(learn, train_cost, old_vali, vali_cost, iter) is True:
                        break
                test_cost = sess.run(cost, feed_dict=self.test_dict)
                print("Test Cost: ", test_cost, "Training Cost: ", train_cost, "Evaluation Cost: ", vali_cost,
                      "Final Learning Rate: ", learn)

class FarSeer:
    # vali = 50, test = 30
    def __init__(self, dataset, num1, num2, drop):
        with tf.name_scope("Basic_Settings"):
            self.D, self.N, self.S = dataset.X, dataset.X['all'].shape[0], dataset.Y
            self.P, self.W, self.B, self.R = {}, {}, {}, {}
            self.train_dict, self.vali_dict, self.test_dict = {}, {}, {}
            self.size_train = num1
            self.size_test = num1 + num2
            self.enc_stack, self.dec_stack, self.pro_stack = {}, 0, 0
            self.drop = drop

    def top_encoder(self, fea, dim, fun):
        self.enc_stack[fea] = 0
        with tf.name_scope("PHD_for_" + fea + "_Training"):
            self.P[fea] = tf.placeholder("float", [None, None])
            self.train_dict[self.P[fea]] = np.split(self.D[fea], [self.size_train, self.size_test, self.N], axis=0)[0]
            self.vali_dict[self.P[fea]] = np.split(self.D[fea], [self.size_train, self.size_test, self.N], axis=0)[1]
            self.test_dict[self.P[fea]] = np.split(self.D[fea], [self.size_train, self.size_test, self.N], axis=0)[2]

        with tf.name_scope(fea + "_Encoder"):
            self.W[fea + '_encT'] = tf.Variable(tf.truncated_normal([self.D[fea].shape[1], dim]))
            self.B[fea + '_encT'] = tf.Variable(tf.truncated_normal([dim]))
            result = tf.nn.dropout(black_magic(tf.add(tf.matmul(self.P[fea], self.W[fea + '_encT']),
                                                      self.B[fea + '_encT']), fun), self.drop)
        return result

    def mid_encoder(self, fea, dim0, dim1, fun, input):
        self.enc_stack[fea] += 1
        with tf.name_scope(fea + "_Encoder_L" + str(self.enc_stack[fea])):
            self.W[fea + '_encT_' + str(self.enc_stack[fea])] = tf.Variable(tf.truncated_normal([dim0, dim1]))
            self.B[fea + '_encT_' + str(self.enc_stack[fea])] = tf.Variable(tf.truncated_normal([dim1]))
            result = tf.nn.dropout(black_magic(tf.add(tf.matmul(input, self.W[fea + '_encT_' + str(self.enc_stack[fea])]),
                                                      self.B[fea + '_encT_' + str(self.enc_stack[fea])]), fun), self.drop)
        return result
    
    def mid_decoder(self, fea, dim0, dim1, fun, input):
        with tf.name_scope(fea + "_Decoder_L" + str(self.dec_stack)):
            self.W[fea + '_decT_' + str(self.dec_stack)] = tf.Variable(tf.truncated_normal([dim0, dim1]))
            self.B[fea + '_decT_' + str(self.dec_stack)] = tf.Variable(tf.truncated_normal([dim1]))
            result = tf.nn.dropout(black_magic(tf.add(tf.matmul(input, self.W[fea + '_decT_' + str(self.dec_stack)]),
                                                      self.B[fea + '_decT_' + str(self.dec_stack)]), fun), self.drop)
        return result

    def bot_decoder(self, enc, fea, dim, fun):
        with tf.name_scope(fea + "_Decoder"):
            self.W[fea + '_decT'] = tf.Variable(tf.truncated_normal([dim, self.D[fea].shape[1]]))
            self.B[fea + '_decT'] = tf.Variable(tf.truncated_normal([self.D[fea].shape[1]]))
            result = tf.nn.dropout(black_magic(tf.add(tf.matmul(enc, self.W[fea + '_decT']),
                                                      self.B[fea + '_decT']), fun), self.drop)
            self.dec_stack = 0
        return result

    def slave_encoder(self, fea, fun, stack, input, weights, bias):
        with tf.name_scope(name=fea + "_EncoderS_L" + str(stack)):
            self.W[fea + '_encS_' + stack] = weights
            self.B[fea + '_encS_' + stack] = bias
            result = tf.nn.dropout(black_magic(tf.add(tf.matmul(input, self.W[fea + '_encS_' + str(stack)]),
                                                      self.B[fea + '_encS_' + str(stack)]), fun), self.drop)
        return result

    def master_encoder(self, fea, fun):
        with tf.name_scope("PHD_for_" + fea + "_Encoding"):
            self.P[fea] = tf.placeholder("float", [None, None])
            self.train_dict[self.P[fea]] = np.split(self.D[fea], [self.size_train, self.size_test, self.N], axis=0)[0]
            self.vali_dict[self.P[fea]] = np.split(self.D[fea], [self.size_train, self.size_test, self.N], axis=0)[1]
            self.test_dict[self.P[fea]] = np.split(self.D[fea], [self.size_train, self.size_test, self.N], axis=0)[2]

        with tf.name_scope(fea + "_EncoderM"):
            self.W[fea + '_encM'] = self.W[fea + '_encT']
            self.B[fea + '_encM'] = self.B[fea + '_encT']
            result = tf.nn.dropout(black_magic(tf.add(tf.matmul(self.P[fea], self.W[fea + '_encM']),
                                                      self.B[fea + '_encM']), fun), self.drop)
            for i in range(1, self.enc_stack[fea] + 1):
                result = self.slave_encoder(result, fun, i, result,
                                            self.W[fea + '_encT_' + str(i)],
                                            self.B[fea + '_encT_' + str(i)])
        return result

    def data_projector(self, target, dim0, dim1, fun):
        self.pro_stack += 1
        with tf.name_scope("Data_Projector" + str(self.pro_stack)):
            self.W['proj' + str(self.pro_stack)] = tf.Variable(tf.truncated_normal([dim0, dim1]))
            self.B['proj' + str(self.pro_stack)] = tf.Variable(tf.truncated_normal([dim1]))
            result = tf.nn.dropout(black_magic(tf.add(tf.matmul(target, self.W['proj' + str(self.pro_stack)]),
                                               self.B['proj' + str(self.pro_stack)]), fun), self.drop)
        return result

    def surv_predictor(self, target, dim, fun):
        with tf.name_scope("PHD_for_Predicting"):
            self.P['surviv'] = tf.placeholder("float", [None, 1])
            self.train_dict[self.P['surviv']] = np.split(self.S, [self.size_train, self.size_test, self.N], axis=0)[0]
            self.vali_dict[self.P['surviv']] = np.split(self.S, [self.size_train, self.size_test, self.N], axis=0)[1]
            self.test_dict[self.P['surviv']] = np.split(self.S, [self.size_train, self.size_test, self.N], axis=0)[2]

        with tf.name_scope("Survivability_Predictor"):
            self.W['surviv'] = tf.get_variable("surviv_W", shape=[dim, 1],
                                               initializer=tf.contrib.layers.xavier_initializer())
            self.B['surviv'] = tf.get_variable("surviv_B", shape=[1],
                                               initializer=tf.contrib.layers.xavier_initializer())
            result = tf.nn.dropout(black_magic(tf.add(tf.matmul(target, self.W['surviv']), 
                                                      self.B['surviv']), fun), self.drop)

            return result

    def mirror_image(self, fea, result, answer, meth, epochs, learn):
        with tf.name_scope("Encoder_Optimizer"):
            cost = tf.reduce_mean(tf.pow(result - answer, 2))
            opti = white_magic(meth, learn, cost)
            old_train, old_vali = 0, 0
            with tf.Session() as sess:
                train_writer = tf.summary.FileWriter("C:/Users/Arthur Keonwoo Kim/PycharmProjects/can2vec/output/",
                                                     sess.graph)
                init = tf.global_variables_initializer()
                sess.run(init)
                for iter in range(epochs):
                    train_cost, _ = sess.run([cost, opti], feed_dict=self.train_dict)
                    vali_cost = sess.run(cost, feed_dict=self.vali_dict)
                    if iter % 1000 == 0:
                        learn = grey_magic(learn, train_cost, old_train)
                        print("Feature: ", fea, iter, "Training Cost: ", train_cost, "Evaluation Cost: ", vali_cost)
                    if red_magic(learn, old_train, train_cost, old_vali, vali_cost, iter) is True:
                        break
                    old_train = train_cost
                test_cost = sess.run(cost, feed_dict=self.test_dict)
                print("Test Cost: ", test_cost, "Training Cost: ", train_cost, "Evaluation Cost: ", vali_cost,
                      "Final Learning Rate: ", learn)

    def foresight(self, result, answer, meth, epochs, learn):
        with tf.name_scope("Survivability_Optimizer"):
            cost = tf.reduce_mean(tf.pow(result - answer, 2))
            opti = white_magic(meth, learn, cost)
            old_train, old_vali = 0, 0
            with tf.Session() as sess:
                train_writer = tf.summary.FileWriter("C:/Users/Arthur Keonwoo Kim/PycharmProjects/can2vec/output/",
                                                     sess.graph)
                init = tf.global_variables_initializer()
                sess.run(init)
                for iter in range(epochs):
                    train_cost, _ = sess.run([cost, opti], feed_dict=self.train_dict)
                    vali_cost = sess.run(cost, feed_dict=self.vali_dict)
                    print(iter, "Training Cost: ", train_cost, "Evaluation Cost: ", vali_cost)
                    if iter % 500 == 0:
                        learn = grey_magic(learn, train_cost, old_train)

                    if red_magic(learn, old_train, train_cost, old_vali, vali_cost, iter) is True:
                        break
                        old_train = train_cost
                test_cost = sess.run(cost, feed_dict=self.test_dict)
                print("Test Cost: ", test_cost, "Training Cost: ", train_cost, "Evaluation Cost: ", vali_cost,
                      "Final Learning Rate: ", learn)








