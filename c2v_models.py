import tensorflow as tf
import numpy as np
from datetime import datetime
from lifelines.utils import _naive_concordance_index

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
    def __init__(self, dataset, drop):
        with tf.name_scope("Basic_Settings"):
            # Needs revamp!
            self.data, self.features = dataset, dataset.F['all']
            self.P, self.W, self.B = {}, {}, {}
            self.item_list = []
            # For feeding dicts and saving variables
            self.train_dict, self.vali_dict, self.test_dict, self.var_dict = {}, {}, {}, {}
            # For stacked encoders and projectors
            self.enc_stack, self.dec_stack = {}, {}
            self.pro_stack = 0
            # Dropout!
            self.drop = drop

    def not_encoder(self, fea):
        with tf.name_scope("PHD_4_" + fea + "_Training"):
            self.P[fea] = tf.placeholder("float", [None, None], name='pholder_' + fea)
            self.train_dict[self.P[fea]] = self.data.T[fea]
            self.vali_dict[self.P[fea]] = self.data.V[fea]
            self.test_dict[self.P[fea]] = self.data.S[fea]
            result = self.P[fea]
        return result

    def top_encoder(self, fea, dim, fun):
        self.enc_stack[fea] = 0
        self.dec_stack[fea] = 0
        with tf.name_scope("PHD_4_" + fea + "_Training"):
            self.P[fea] = tf.placeholder("float", [None, None], name='pholder_' + fea)
            self.train_dict[self.P[fea]] = self.data.T[fea]
            self.vali_dict[self.P[fea]] = self.data.V[fea]
            self.test_dict[self.P[fea]] = self.data.S[fea]

        with tf.name_scope(fea + "_Encoder"):
            name = fea + '_encT'
            self.W[name] = tf.get_variable(name='W_' + name, shape=[dim, self.data.F[fea]],
                                           initializer=tf.contrib.layers.xavier_initializer())
            self.var_dict['W_' + name] = self.W[name]
            self.B[name] = tf.get_variable(name='B_' + name, shape=[dim, 1],
                                           initializer=tf.contrib.layers.xavier_initializer())
            self.var_dict['B_' + name] = self.B[name]
            result = tf.nn.dropout(black_magic(tf.add(tf.matmul(self.W[name], self.P[fea], name='mul_' + name),
                                                      self.B[name], name='add_' + name), fun),
                                   self.drop, name='drop_' + name)
            tf.summary.histogram('W_' + name, self.W[name], collections=[fea])
            tf.summary.histogram('W_' + name, self.W[name], collections=['main'])
            tf.summary.histogram('B_' + name, self.B[name], collections=[fea])
            tf.summary.histogram('B_' + name, self.B[name], collections=['main'])
        return result

    def mid_encoder(self, fea, dim, fun, target):
        self.enc_stack[fea] += 1
        with tf.name_scope(fea + "_Encoder_L" + str(self.enc_stack[fea])):
            name = fea + '_encT_' + str(self.enc_stack[fea])
            self.W[name] = tf.get_variable(name='W_' + name, shape=[dim, target.get_shape()[0]],
                                           initializer=tf.contrib.layers.xavier_initializer())
            self.var_dict['W_' + name] = self.W[name]
            self.B[name] = tf.get_variable(name='B_' + name, shape=[dim, 1],
                                           initializer=tf.contrib.layers.xavier_initializer())
            self.var_dict['B_' + name] = self.B[name]
            result = tf.nn.dropout(black_magic(tf.add(tf.matmul(self.W[name], target, name='mul_' + name),
                                                      self.B[name], name='add_' + name), fun),
                                   self.drop, name='drop_' + name)
            tf.summary.histogram('W_' + name, self.W[name], collections=[fea])
            tf.summary.histogram('W_' + name, self.W[name], collections=['main'])
            tf.summary.histogram('B_' + name, self.B[name], collections=[fea])
            tf.summary.histogram('B_' + name, self.B[name], collections=['main'])
        return result
    
    def mid_decoder(self, fea, dim, fun, target):
        self.dec_stack[fea] += 1
        with tf.name_scope(fea + "_Decoder_L" + str(self.dec_stack[fea])):
            name = fea + '_decT_' + str(self.dec_stack[fea])
            self.W[name] = tf.get_variable(name='W_'+name, shape=[dim, target.get_shape()[0]],
                                           initializer=tf.contrib.layers.xavier_initializer())
            self.B[name] = tf.get_variable(name='B_'+name, shape=[dim, 1],
                                           initializer=tf.contrib.layers.xavier_initializer())
            result = tf.nn.dropout(black_magic(tf.add(tf.matmul(self.W[name], target, name='mul_' + name),
                                                      self.B[name], name='add_' + name), fun),
                                   self.drop, name='drop_' + name)

            tf.summary.histogram('W_' + name, self.W[name], collections=[fea])
            tf.summary.histogram('B_' + name, self.B[name], collections=[fea])
        return result

    def bot_decoder(self, fea, fun, target):
        with tf.name_scope(fea + "_Decoder"):
            name = fea + '_decT'
            self.W[name] = tf.get_variable(name='W_'+name, shape=[self.data.F[fea], target.get_shape()[0]],
                                           initializer=tf.contrib.layers.xavier_initializer())
            self.B[name] = tf.get_variable(name='B_'+name, shape=[self.data.F[fea], 1],
                                           initializer=tf.contrib.layers.xavier_initializer())
            result = black_magic(tf.add(tf.matmul(self.W[name], target, name='mul_' + name),
                                        self.B[name], name='add_' + name), fun)
            tf.summary.histogram('W_' + name, self.W[name], collections=[fea])
            tf.summary.histogram('B_' + name, self.B[name], collections=[fea])
        return result

    def data_projector(self, target, dim1, dim0, fun):
        self.pro_stack += 1
        with tf.name_scope("Data_Projector_" + str(self.pro_stack)):
            name = 'proj_' + str(self.pro_stack)
            self.W[name] = tf.get_variable(name='W_'+name, shape=[dim0, dim1],
                                               initializer=tf.contrib.layers.xavier_initializer())
            self.B[name] = tf.get_variable(name='B_'+name, shape=[dim0, 1],
                                               initializer=tf.contrib.layers.xavier_initializer())
            result = tf.nn.dropout(black_magic(tf.add(tf.matmul(self.W[name], target, name='mul_' + name),
                                                      self.B[name], name='add_' + name), fun),
                                   self.drop, name='drop_' + name)
            tf.summary.histogram('W_' + name, self.W[name], ['main'])
            tf.summary.histogram('B_' + name, self.B[name], ['main'])
        return result
    
    def surv_predictor(self, target, fun):
        with tf.name_scope("PHD_4_Prediction"):
            self.P['sur'] = tf.placeholder("float", [1, None])
            self.train_dict[self.P['sur']] = self.data.T['sur']
            self.vali_dict[self.P['sur']] = self.data.V['sur']
            self.test_dict[self.P['sur']] = self.data.S['sur']

        with tf.name_scope("Survivability_Predictor"):
            self.W['sur'] = tf.get_variable("W_sur", shape=[1, target.get_shape()[0]],
                                               initializer=tf.contrib.layers.xavier_initializer())
            self.B['sur'] = tf.get_variable("B_sur", shape=[1, 1],
                                               initializer=tf.contrib.layers.xavier_initializer())
            result = black_magic(tf.add(tf.matmul(self.W['sur'], target, name='mul_sur'),
                                        self.B['sur'], name='add_sur'), fun)
            tf.summary.histogram('W_sur', self.W['sur'], ['main'])
            tf.summary.histogram('B_sur', self.B['sur'], ['main'])
        return result

    def cox_cummulative(self, target):
        target = tf.reverse(target, [-1], name='reverse_cox')
        target = tf.exp(target, name='exp_cox')
        target = tf.slice(target, [0, 0], [-1, self.data.T['sur'].shape[1]], name='slice_cox')
        values = tf.split(target, target.get_shape()[1], 1, name='split_cox')
        csum = tf.zeros_like(values[0], name='zeros_cox') + 1
        out = []
        for val in values:
            out.append(csum)
            csum = tf.add(csum, val, name='add_cox')
        result = tf.reverse(tf.concat(out, 1, name='concat_cox'), [-1], name='reverse2_cox')
        return result

    def re_constructor(self, target, dim, fun):
        with tf.name_scope("PHD_4_Reconstruction"):
            self.P['recon'] = tf.placeholder("float", [None, self.data.features])
            self.train_dict[self.P['recon']] = self.data.T['all']
            self.vali_dict[self.P['recon']] = self.data.V['all']
            self.test_dict[self.P['recon']] = self.data.S['all']

        with tf.name_scope("Data_Reconstructor"):
            self.W['recon'] = tf.get_variable("W_recon", shape=[self.data.features, dim],
                                              initializer=tf.contrib.layers.xavier_initializer())
            tf.summary.histogram('W_recon', self.W['recon'], ['main'])
            self.B['recon'] = tf.get_variable("B_recon", shape=[self.data.features, 1],
                                              initializer=tf.contrib.layers.xavier_initializer())
            tf.summary.histogram('B_recon', self.B['recon'], ['main'])
            result = black_magic(tf.add(tf.matmul(self.W['recon'], target, name='mul_recon'),
                                        self.B['recon'], name='add_recon'), fun)
            return result

    def optimize_AEncoders(self):
        with tf.name_scope("Train_AEncoders"):
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            with tf.Session(config=config) as sess:
                train_writer = tf.summary.FileWriter("./PHASE1/" + str(datetime.now()), sess.graph)
                init = tf.global_variables_initializer()
                saver = tf.train.Saver(self.var_dict)
                sess.run(init)
                self.begin_training(self.item_list, sess, train_writer)
                saver.save(sess, "./saved/model_step1.ckpt")

    def begin_training(self, split_opts, sess, tw):
        with tf.name_scope("Begin_Training"):
            for opt in split_opts:
                tf.summary.scalar('cost_' + opt.fea, opt.cost, collections=[opt.fea])
                merged = tf.summary.merge_all(opt.fea)
                old_train, old_vali = 0, 0
                for iter in range(opt.epochs):
                    train_cost, _, summ = sess.run([opt.cost, opt.opti, merged], feed_dict=self.train_dict)
                    valid_cost = sess.run(opt.cost, feed_dict=self.vali_dict)
                    opt.learn = grey_magic(opt.learn, train_cost, old_train)
                    if red_magic(opt.learn, old_train, train_cost, old_vali, valid_cost, iter, opt.epochs) is True:
                        break
                    tw.add_summary(summ, iter)
                    old_train = train_cost
                test_cost = sess.run(opt.cost, feed_dict=self.test_dict)
                print("Feature: ", opt.fea, "Test Cost: ", test_cost, "Training Cost: ", train_cost, "Evaluation Cost: "
                      , valid_cost, "Final Learning Rate: ", opt.learn)

    def optimize_SPredictor(self, result, meth, epochs, learn):
        with tf.name_scope("Train_SPredictor"):
            cost = tf.reduce_mean(tf.pow(result - self.P['sur'], 2))
            opti = white_magic(meth, learn, cost)
            tf.summary.scalar('cost_survivability', cost, collections=['main'])
            merged = tf.summary.merge_all('main')
            old_train, old_vali = 0, 0
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            with tf.Session(config=config) as sess:
                init = tf.global_variables_initializer()
                saver = tf.train.Saver(self.var_dict)
                train_writer = tf.summary.FileWriter("./PHASE2/" + str(datetime.now()), sess.graph,)
                sess.run(init)
                saver.restore(sess, "./saved/model_step1.ckpt")
                for iter in range(epochs):
                    train_cost, _, summ, surv_pred, surv_real = sess.run([cost, opti, merged, result, self.P['sur']],
                                                                         feed_dict=self.train_dict)
                    valid_cost, valid_pred, valid_real = sess.run([cost, result, self.P['sur']],
                                                                  feed_dict=self.vali_dict)
                    r, p, o = surv_real, surv_pred, self.data.T['cen']
                    r2, p2, o2 =valid_real, valid_pred, self.data.V['cen']
                    print("Cost for training session", train_cost)
                    print("C-Index for training session", _naive_concordance_index(r[0], p[0], o[0]))
                    print("Cost for validation session", valid_cost)
                    print("C-Index for validation session", _naive_concordance_index(r2[0], p2[0], o2[0]))
                    learn = grey_magic(learn, train_cost, old_train)
                    if red_magic(learn, old_train, train_cost, old_vali, valid_cost, iter, epochs) is True:
                        break
                    old_train = train_cost
                    train_writer.add_summary(summ, iter)
                test_cost, test_pred, test_real = sess.run([cost, result, self.P['sur']], feed_dict=self.test_dict)
                r3, p3, o3 = test_real, test_pred, self.data.S['cen']
                print("Survivability Predictor Test Cost: ", test_cost, "Training Cost: ", train_cost,
                      "Evaluation Cost: ", valid_cost, "C-Index for test session: ",
                      _naive_concordance_index(r3[0], p3[0], o3[0]), "Final Learning Rate: ", learn)
                saver.save(sess, "./saved/survival_regression.ckpt")

    def optimize_CPredictor(self, result, epochs, learn):
        with tf.name_scope("Train_CPredictor"):
            merged = tf.summary.merge_all('main')
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            with tf.Session(config=config) as sess:
                init = tf.global_variables_initializer()
                saver = tf.train.Saver(self.var_dict)
                train_writer = tf.summary.FileWriter("./PHASE2/" + str(datetime.now()), sess.graph,)
                sess.run(init)
                saver.restore(sess, "./saved/model_step1.ckpt")
                for iter in range(epochs):
                    partial_sum = self.cox_cummulative(result)
                    final_sum = tf.subtract(tf.log(result + 1, name='log1_cox'),
                                            tf.log(partial_sum + 1, name='log2_cox'),
                                            name='sub_cox')
                    final_product = final_sum * self.data.T['cen']
                    cost = -tf.reduce_sum(final_product)
                    opti = white_magic('grad', learn, cost)
                    c, _, summ, surv_pred, surv_real= sess.run([cost, opti, merged,result, self.P['sur']],
                                                                feed_dict=self.train_dict)
                    print("Likelihood function value: ", c)
                    valid_pred, valid_real = sess.run([result, self.P['sur']], feed_dict=self.vali_dict)
                    r, p, o = surv_real, surv_pred, self.data.T['cen']
                    r2, p2, o2 = valid_real, valid_pred, self.data.V['cen']
                    print("C-Index for training session", _naive_concordance_index(r[0], p[0], o[0]))
                    print("C-Index for validation session", _naive_concordance_index(r2[0], p2[0], o2[0]))
                    train_writer.add_summary(summ, iter)
                test_pred, test_real = sess.run([result, self.P['sur']], feed_dict=self.test_dict)
                r3, p3, o3 = test_real, test_pred, self.data.S['cen']
                print("C-Index for test session", _naive_concordance_index(r3[0], p3[0], o3[0]))
                saver.save(sess, "./saved/cox_regression.ckpt")

    def optimize_RConstructor(self, result, meth, epochs, learn):
        with tf.name_scope("Reconstruction_Optimizer"):
            cost = tf.reduce_mean(tf.pow(result - self.P['recon'], 2))
            opti = white_magic(meth, learn, cost)
            tf.summary.scalar('cost_reconstruction', cost, collections=['main'])
            merged = tf.summary.merge_all('main')
            old_train, old_vali = 0, 0
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            with tf.Session(config=config) as sess:
                init = tf.global_variables_initializer()
                saver = tf.train.Saver(self.var_dict)
                train_writer = tf.summary.FileWriter("./PHASE3/" + str(datetime.now()), sess.graph, )
                sess.run(init)
                saver.restore(sess, "./saved/data_reconstruction.ckpt")
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

class SplitOptimizer:
    def __init__(self, fea, result, answer, meth, epochs, learn):
        with tf.name_scope("Split_Optimizer_" + fea):
            self.fea, self.result, self.meth, self.epochs, self.learn = fea, result, meth, epochs, learn
            self.cost = tf.reduce_mean(tf.pow(result - answer, 2))
            self.opti = white_magic(meth, learn, self.cost)

