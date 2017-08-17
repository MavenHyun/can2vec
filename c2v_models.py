import tensorflow as tf
import numpy as np
from datetime import datetime
import time
import train_set as ts

#   Select your spell for activation function.
def black_magic(operation, name):
    black_spell = {
        'relu': tf.nn.relu(operation),
        'sigmoid': tf.nn.sigmoid(operation),
        'tanh': tf.nn.tanh(operation),
        'raw': operation
    }
    return black_spell[name]

#   Select your spell for optimization.
def white_magic(name, learn, cost):
    white_spell = {
        'adam': tf.train.AdamOptimizer,
        'rms': tf.train.RMSPropOptimizer,
        'adag': tf.train.AdagradOptimizer,
        'adad': tf.train.AdadeltaOptimizer,
        'grad': tf.train.GradientDescentOptimizer
    }
    return white_spell[name](learn).minimize(cost)

#  Learning Rate decay, will consider one of the tf modules.
def grey_magic(learn, old_train, new_train):
    if 0 <= old_train - new_train < 0.001:
        mod_learn = learn * 0.3
    else:
        mod_learn = learn
    return mod_learn

#  Automatically terminates training session
def red_magic(learn, old_train, new_train, old_vali, new_vali, iter, epochs):
        # Halt this model when it produces the most optimal results.
    if abs(new_train - new_vali) < 0.1 and new_train < 1 and new_vali < 1 and iter > (epochs * 0.4):
        return True
        #  No progress, no reason to learn.
    elif 0 <= old_vali - new_vali < 0.000001 and 0 <= old_train - new_train < 0.000001 and iter > (epochs * 0.4):
        return True
        #  Learning rate threshold
    elif learn < 1.0e-30:
        return True
    else:
        return False

# Still needs improvement!
class FarSeer:
    # vali = 50, test = 30
    def __init__(self, dataset, num1, num2, drop):
        with tf.name_scope("Basic_Settings"):
            self.D, self.N, self.S = dataset.X, dataset.X['all'].shape[0], dataset.Y
            self.P, self.W, self.B = {}, {}, {}
            self.train_dict, self.vali_dict, self.test_dict, self.answer_dict = {}, {}, {}, {}
            self.size_train, self.size_test = num1, num1 + num2
            # For stacked encoders and projectors
            self.enc_stack, self.dec_stack, self.pro_stack = {}, {}, 0
            # A list of costs and optimizers
            self.item_list = []
            # Dropout!
            self.drop = drop

    def not_encoder(self, fea):
        with tf.name_scope("PHD_4_" + fea + "_Training"):
            self.P[fea] = tf.placeholder("float", [None, None])
            self.train_dict[self.P[fea]] = np.split(self.D[fea], [self.size_train, self.size_test, self.N], axis=0)[0]
            self.vali_dict[self.P[fea]] = np.split(self.D[fea], [self.size_train, self.size_test, self.N], axis=0)[1]
            self.test_dict[self.P[fea]] = np.split(self.D[fea], [self.size_train, self.size_test, self.N], axis=0)[2]
            result = self.P[fea]
        return result

    def top_encoder(self, fea, dim, fun):
        self.enc_stack[fea] = 0
        self.dec_stack[fea] = 0
        with tf.name_scope("PHD_4_" + fea + "_Training"):
            self.P[fea] = tf.placeholder("float", [None, None])
            self.train_dict[self.P[fea]] = np.split(self.D[fea], [self.size_train, self.size_test, self.N], axis=0)[0]
            self.vali_dict[self.P[fea]] = np.split(self.D[fea], [self.size_train, self.size_test, self.N], axis=0)[1]
            self.test_dict[self.P[fea]] = np.split(self.D[fea], [self.size_train, self.size_test, self.N], axis=0)[2]

        with tf.name_scope(fea + "_Encoder"):
            name = fea + '_encT'
            self.W[name] = tf.get_variable(name='W_' + name, shape=[self.D[fea].shape[1], dim],
                                           initializer=tf.contrib.layers.xavier_initializer())
            tf.summary.histogram('W_' + name, self.W[name], collections=[fea])
            tf.summary.histogram('W_' + name, self.W[name], collections=['main'])
            self.B[name] = tf.get_variable(name='B_' + name, shape=[dim],
                                           initializer=tf.contrib.layers.xavier_initializer())
            tf.summary.histogram('B_' + name, self.B[name], collections=[fea])
            tf.summary.histogram('B_' + name, self.B[name], collections=['main'])
            result = tf.nn.dropout(black_magic(tf.add(tf.matmul(self.P[fea], self.W[name]), self.B[name]), fun),
                                   self.drop)
        return result

    def mid_encoder(self, fea, dim0, dim1, fun, input):
        self.enc_stack[fea] += 1
        with tf.name_scope(fea + "_Encoder_L" + str(self.enc_stack[fea])):
            name = fea + '_encT_' + str(self.enc_stack[fea])
            self.W[name] = tf.get_variable(name='W_' + name, shape=[dim0, dim1],
                                           initializer=tf.contrib.layers.xavier_initializer())
            tf.summary.histogram('W_' + name, self.W[name], collections=[fea])
            tf.summary.histogram('W_' + name, self.W[name], collections=['main'])
            self.B[name] = tf.get_variable(name='B_' + name, shape=[dim1],
                                           initializer=tf.contrib.layers.xavier_initializer())
            tf.summary.histogram('B_' + name, self.B[name], collections=[fea])
            tf.summary.histogram('B_' + name, self.B[name], collections=['main'])
            result = tf.nn.dropout(black_magic(tf.add(tf.matmul(input, self.W[name]), self.B[name]), fun),
                                   self.drop)
        return result
    
    def mid_decoder(self, fea, dim0, dim1, fun, input):
        self.dec_stack[fea] += 1
        with tf.name_scope(fea + "_Decoder_L" + str(self.dec_stack[fea])):
            name = fea + '_decT_' + str(self.dec_stack[fea])
            self.W[name] = tf.get_variable(name='W_'+name, shape=[dim0, dim1], 
                                           initializer=tf.contrib.layers.xavier_initializer())
            tf.summary.histogram('W_' + name, self.W[name], collections=[fea])
            self.B[name] = tf.get_variable(name='B_'+name, shape=[dim1], 
                                           initializer=tf.contrib.layers.xavier_initializer())
            tf.summary.histogram('B_' + name, self.B[name], collections=[fea])
            result = tf.nn.dropout(black_magic(tf.add(tf.matmul(input, self.W[name]), self.B[name]), fun),
                                   self.drop)
        return result

    def bot_decoder(self, enc, fea, dim, fun):
        with tf.name_scope(fea + "_Decoder"):
            name = fea + '_decT'
            self.W[name] = tf.get_variable(name='W_'+name, shape=[dim, self.D[fea].shape[1]],
                                           initializer=tf.contrib.layers.xavier_initializer())
            tf.summary.histogram('W_'+name, self.W[name], collections=[fea])
            self.B[name] = tf.get_variable(name='B_'+name, shape=[self.D[fea].shape[1]],
                                           initializer=tf.contrib.layers.xavier_initializer())
            tf.summary.histogram('B_'+name, self.B[name], collections=[fea])
            result = black_magic(tf.add(tf.matmul(enc, self.W[name]), self.B[name]), fun)
        return result

    def data_projector(self, target, dim0, dim1, fun):
        self.pro_stack += 1
        with tf.name_scope("Data_Projector_" + str(self.pro_stack)):
            name = 'proj_' + str(self.pro_stack)
            self.W[name] = tf.get_variable(name='W_'+name, shape=[dim0, dim1],
                                               initializer=tf.contrib.layers.xavier_initializer())
            tf.summary.histogram('W_'+name,  self.W[name], ['main'])
            self.B[name] = tf.get_variable(name='B_'+name, shape=[dim1],
                                               initializer=tf.contrib.layers.xavier_initializer())
            tf.summary.histogram('B_'+name,  self.B[name], ['main'])
            result = tf.nn.dropout(black_magic(tf.add(tf.matmul(target, self.W[name]),
                                               self.B[name]), fun), self.drop)
        return result
    '''
        def data_reconstructor(self, target, dim, fun):
        with tf.name_scope("PHD_for_Reconstruction"):
            self.P['recon'] = tf.placeholder("float", [None, 1])
            self.train_dict[self.P['recon']] = np.split(self.S, [self.size_train, self.size_test, self.N], axis=0)[0]
            self.vali_dict[self.P['recon']] = np.split(self.S, [self.size_train, self.size_test, self.N], axis=0)[1]
            self.test_dict[self.P['recon']] = np.split(self.S, [self.size_train, self.size_test, self.N], axis=0)[2]

        with tf.name_scope("Data_Reconstructor"):
            self.W['recon'] = tf.get_variable("recon_W", shape=[dim, self.N],
                                               initializer=tf.contrib.layers.xavier_initializer())
            tf.summary.histogram('weights_pred', self.W['recon'], ['recon'])
            self.B['recon'] = tf.get_variable("recon_B", shape=[self.N],
                                               initializer=tf.contrib.layers.xavier_initializer())
            tf.summary.histogram('bias_pred', self.B['recon'], ['recon'])
            result = black_magic(tf.add(tf.matmul(target, self.W['recon']), self.B['recon']), fun)
            return result

    '''
    def surv_predictor(self, target, dim, fun):
        with tf.name_scope("PHD_for_Prediction"):
            self.P['surviv'] = tf.placeholder("float", [None, 1])
            self.train_dict[self.P['surviv']] = np.split(self.S, [self.size_train, self.size_test, self.N], axis=0)[0]
            self.vali_dict[self.P['surviv']] = np.split(self.S, [self.size_train, self.size_test, self.N], axis=0)[1]
            self.test_dict[self.P['surviv']] = np.split(self.S, [self.size_train, self.size_test, self.N], axis=0)[2]

        with tf.name_scope("Survivability_Predictor"):
            self.W['surviv'] = tf.get_variable("surviv_W", shape=[dim, 1],
                                               initializer=tf.contrib.layers.xavier_initializer())
            tf.summary.histogram('weights_pred', self.W['surviv'], ['main'])
            self.B['surviv'] = tf.get_variable("surviv_B", shape=[1],
                                               initializer=tf.contrib.layers.xavier_initializer())
            tf.summary.histogram('bias_pred', self.B['surviv'], ['main'])
            result = black_magic(tf.add(tf.matmul(target, self.W['surviv']), self.B['surviv']), fun)
            return result

    def optimize_AEncoders(self):
        with tf.name_scope("Train_AEncoders"):
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            with tf.Session(config=config) as sess:
                train_writer = tf.summary.FileWriter("./phase1/" + str(datetime.now()), sess.graph)
                init = tf.global_variables_initializer()
                saver = tf.train.Saver()
                sess.run(init)
                self.begin_training(self.item_list, sess, train_writer)
                saver.save(sess, "/tmp/model_step1.ckpt")

    def begin_training(self, split_opts, sess, tw):
        with tf.name_scope("Begin_Training"):
            for opt in split_opts:
                tf.summary.scalar('cost_' + opt.fea, opt.cost, collections=[opt.fea])
                merged = tf.summary.merge_all(opt.fea)
                old_train, old_vali = 0, 0
                for iter in range(opt.epochs):
                    train_cost, _, summ = sess.run([opt.cost, opt.opti, merged], feed_dict=self.train_dict)
                    vali_cost = sess.run(opt.cost, feed_dict=self.vali_dict)
                    if iter % 100 == 0:
                        opt.learn = grey_magic(opt.learn, train_cost, old_train)
                        print("Feature: ", opt.fea, iter, "Training Cost: ", train_cost, "Evaluation Cost: ", vali_cost)
                    if red_magic(opt.learn, old_train, train_cost, old_vali, vali_cost, iter, opt.epochs) is True:
                        break
                    tw.add_summary(summ, iter)
                    old_train = train_cost
                test_cost = sess.run(opt.cost, feed_dict=self.test_dict)
                print("Feature: ", opt.fea, "Test Cost: ", test_cost, "Training Cost: ", train_cost, "Evaluation Cost: ",
                      vali_cost, "Final Learning Rate: ", opt.learn)

    def optimize_SPredictor(self, result, answer, meth, epochs, learn):
        with tf.name_scope("Train_SPredictor"):
            cost = tf.reduce_mean(tf.pow(result - answer, 2))
            opti = white_magic(meth, learn, cost)
            tf.summary.scalar('cost_survivability', cost, collections=['main'])
            merged = tf.summary.merge_all('main')
            old_train, old_vali = 0, 0
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            with tf.Session(config=config) as sess:
                init = tf.global_variables_initializer()
                saver = tf.train.Saver()
                train_writer = tf.summary.FileWriter("./phase2/" + str(datetime.now()), sess.graph,)
                sess.run(init)
                saver.restore(sess, "/tmp/model_step1.ckpt")
                for iter in range(epochs):
                    train_cost, _, summ = sess.run([cost, opti, merged], feed_dict=self.train_dict)
                    vali_cost = sess.run(cost, feed_dict=self.vali_dict)
                    if iter % 100 == 0:
                        print(iter, "Training Cost: ", train_cost, "Evaluation Cost: ", vali_cost)
                        learn = grey_magic(learn, train_cost, old_train)
                    if red_magic(learn, old_train, train_cost, old_vali, vali_cost, iter, epochs) is True:
                        break
                    old_train = train_cost
                    train_writer.add_summary(summ, iter)
                test_cost = sess.run(cost, feed_dict=self.test_dict)
                print("Survivability Predictor Test Cost: ", test_cost, "Training Cost: ", train_cost,
                      "Evaluation Cost: ", vali_cost, "Final Learning Rate: ", learn)
    '''
        def reanimate(self, result, answer, meth, epochs, learn):
        with tf.name_scope("Reconstruction_Optimizer"):
            cost = tf.reduce_mean(tf.pow(result - answer, 2))
            opti = white_magic(meth, learn, cost)
            old_train, old_vali = 0, 0
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            with tf.Session(config=config) as sess:
                train_writer = tf.summary.FileWriter("C:/Users/Arthur Keonwoo Kim/PycharmProjects/can2vec/output/",
                                                     sess.graph)
                init = tf.global_variables_initializer()
                sess.run(init)
                for iter in range(epochs):
                    train_cost, _ = sess.run([cost, opti], feed_dict=self.train_dict)
                    vali_cost = sess.run(cost, feed_dict=self.vali_dict)
                    print(iter, "Training Cost: ", train_cost, "Evaluation Cost: ", vali_cost)
                    if iter % 100 == 0:
                        learn = grey_magic(learn, train_cost, old_train)
                    if red_magic(learn, old_train, train_cost, old_vali, vali_cost, iter, epochs) is True:
                        break
                    old_train = train_cost
                test_cost = sess.run(cost, feed_dict=self.test_dict)
                print("Test Cost: ", test_cost, "Training Cost: ", train_cost, "Evaluation Cost: ", vali_cost,
                      "Final Learning Rate: ", learn)
    '''
class SplitOptimizer:
    def __init__(self, fea, result, answer, meth, epochs, learn):
        with tf.name_scope("Split_Optimizer_" + fea):
            self.fea, self.result, self.meth, self.epochs, self.learn = fea, result, meth, epochs, learn
            self.cost = tf.reduce_mean(tf.pow(result - answer, 2))
            self.opti = white_magic(meth, learn, self.cost)
