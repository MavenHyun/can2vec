import tensorflow as tf
import numpy as np
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
        'adam': tf.train.AdamOptimizer(learn).minimize(cost),
        'rms': tf.train.RMSPropOptimizer(learn).minimize(cost),
        'adag': tf.train.AdagradOptimizer(learn).minimize(cost),
        'adad': tf.train.AdadeltaOptimizer(learn).minimize(cost),
        'grad': tf.train.GradientDescentOptimizer(learn).minimize(cost)
    }
    return white_spell[name]

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
        with tf.name_scope("PHD_for_" + fea + "_Training"):
            self.P[fea] = tf.placeholder("float", [None, None])
            self.train_dict[self.P[fea]] = np.split(self.D[fea], [self.size_train, self.size_test, self.N], axis=0)[0]
            self.vali_dict[self.P[fea]] = np.split(self.D[fea], [self.size_train, self.size_test, self.N], axis=0)[1]
            self.test_dict[self.P[fea]] = np.split(self.D[fea], [self.size_train, self.size_test, self.N], axis=0)[2]
            result = self.P[fea]
        return result

    def top_encoder(self, fea, dim, fun):
        self.enc_stack[fea] = 0
        self.dec_stack[fea] = 0
        with tf.name_scope("PHD_for_" + fea + "_Training"):
            self.P[fea] = tf.placeholder("float", [None, None])
            self.train_dict[self.P[fea]] = np.split(self.D[fea], [self.size_train, self.size_test, self.N], axis=0)[0]
            self.vali_dict[self.P[fea]] = np.split(self.D[fea], [self.size_train, self.size_test, self.N], axis=0)[1]
            self.test_dict[self.P[fea]] = np.split(self.D[fea], [self.size_train, self.size_test, self.N], axis=0)[2]

        with tf.name_scope(fea + "_Encoder"):
            name = fea + '_encT'
            self.W[name] = tf.get_variable(name='W_'+name, shape=[self.D[fea].shape[1], dim],
                                           initializer=tf.contrib.layers.xavier_initializer())
            tf.summary.histogram('W_'+name, self.W[name])
            self.B[name] = tf.get_variable(name='B_'+name, shape=[dim],
                                           initializer=tf.contrib.layers.xavier_initializer())
            tf.summary.histogram('B_'+name, self.B[name])
            result = tf.nn.dropout(black_magic(tf.add(tf.matmul(self.P[fea], self.W[name]), self.B[name]), fun),
                                   self.drop)
        return result

    def mid_encoder(self, fea, dim0, dim1, fun, input):
        self.enc_stack[fea] += 1
        with tf.name_scope(fea + "_Encoder_L" + str(self.enc_stack[fea])):
            name = fea + '_encT_' + str(self.enc_stack[fea])
            self.W[name] = tf.get_variable(name='W_'+name, shape=[dim0, dim1], 
                                           initializer=tf.contrib.layers.xavier_initializer())
            tf.summary.histogram('W_'+name, self.W[name])
            self.B[name] = tf.get_variable(name='B_'+name, shape=[dim1], 
                                           initializer=tf.contrib.layers.xavier_initializer())
            tf.summary.histogram('B_'+name, self.B[name])
            result = tf.nn.dropout(black_magic(tf.add(tf.matmul(input, self.W[name]), self.B[name]), fun), self.drop)
        return result
    
    def mid_decoder(self, fea, dim0, dim1, fun, input):
        self.dec_stack[fea] += 1
        with tf.name_scope(fea + "_Decoder_L" + str(self.dec_stack[fea])):
            name = fea + '_decT_' + str(self.dec_stack[fea])
            self.W[name] = tf.get_variable(name='W_'+name, shape=[dim0, dim1], 
                                           initializer=tf.contrib.layers.xavier_initializer())
            tf.summary.histogram('W_'+name, self.W[name])
            self.B[name] = tf.get_variable(name='B_'+name, shape=[dim1], 
                                           initializer=tf.contrib.layers.xavier_initializer())
            tf.summary.histogram('B_'+name, self.B[name])
            result = tf.nn.dropout(black_magic(tf.add(tf.matmul(input, self.W[name]), self.B[name]), fun), self.drop)
        return result

    def bot_decoder(self, enc, fea, dim, fun):
        with tf.name_scope(fea + "_Decoder"):
            name = fea + '_decT'
            self.W[name] = tf.get_variable(name='W_'+name, shape=[dim, self.D[fea].shape[1]],
                                           initializer=tf.contrib.layers.xavier_initializer())
            tf.summary.histogram('W_'+name, self.W[name])
            self.B[name] = tf.get_variable(name='B_'+name, shape=[self.D[fea].shape[1]],
                                           initializer=tf.contrib.layers.xavier_initializer())
            tf.summary.histogram('B_'+name, self.B[name])
            result = black_magic(tf.add(tf.matmul(enc, self.W[name]), self.B[name]), fun)
        return result

    def slave_encoder(self, fea, fun, stack, input, weights, bias):
        with tf.name_scope(fea + "_EncoderS_L" + str(stack)):
            self.W[fea + '_encS_' + str(stack)] = weights
            tf.summary.histogram('weights_encS_L' + str(stack) + '_' + fea, weights)
            self.B[fea + '_encS_' + str(stack)] = bias
            tf.summary.histogram('bias_encS_L' + str(stack) + '_' + fea, bias)
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
            tf.summary.histogram('weights_encM_' + fea, self.W[fea + '_encM'])
            self.B[fea + '_encM'] = self.B[fea + '_encT']
            tf.summary.histogram('bias_encM_' + fea, self.B[fea + '_encM'])
            result = black_magic(tf.add(tf.matmul(self.P[fea], self.W[fea + '_encM']), self.B[fea + '_encM']), fun)
            for i in range(1, self.enc_stack[fea] + 1):
                result = self.slave_encoder(fea, fun, i, result,
                                            self.W[fea + '_encT_' + str(i)],
                                            self.B[fea + '_encT_' + str(i)])
        return result

    def the_alchemist(self, name, target, dim0, dim1, fun):
        with tf.name_scope("The_Alchemist_of_" + name):
            self.W['proj' + str(self.pro_stack)] = tf.Variable(tf.truncated_normal([dim0, dim1]))
            tf.summary.histogram('weights_proj' + str(self.pro_stack),  self.W['proj' + str(self.pro_stack)])
            self.B['proj' + str(self.pro_stack)] = tf.Variable(tf.truncated_normal([dim1]))
            tf.summary.histogram('bias_proj' + str(self.pro_stack),  self.B['proj' + str(self.pro_stack)])
            result = tf.nn.dropout(black_magic(tf.add(tf.matmul(target, self.W['proj' + str(self.pro_stack)]),
                                               self.B['proj' + str(self.pro_stack)]), fun), self.drop)
        return result

    def data_reconstructor(self, target, dim, fun):
        with tf.name_scope("PHD_for_Reconstruction"):
            self.P['recon'] = tf.placeholder("float", [None, 1])
            self.train_dict[self.P['recon']] = np.split(self.S, [self.size_train, self.size_test, self.N], axis=0)[0]
            self.vali_dict[self.P['recon']] = np.split(self.S, [self.size_train, self.size_test, self.N], axis=0)[1]
            self.test_dict[self.P['recon']] = np.split(self.S, [self.size_train, self.size_test, self.N], axis=0)[2]

        with tf.name_scope("Data_Reconstructor"):
            self.W['recon'] = tf.get_variable("recon_W", shape=[dim, self.N],
                                               initializer=tf.contrib.layers.xavier_initializer())
            tf.summary.histogram('weights_pred', self.W['recon'])
            self.B['recon'] = tf.get_variable("recon_B", shape=[self.N],
                                               initializer=tf.contrib.layers.xavier_initializer())
            tf.summary.histogram('bias_pred', self.B['recon'])
            result = black_magic(tf.add(tf.matmul(target, self.W['recon']), self.B['recon']), fun)
            return result

    def surv_predictor(self, target, dim, fun):
        with tf.name_scope("PHD_for_Prediction"):
            self.P['surviv'] = tf.placeholder("float", [None, 1])
            self.train_dict[self.P['surviv']] = np.split(self.S, [self.size_train, self.size_test, self.N], axis=0)[0]
            self.vali_dict[self.P['surviv']] = np.split(self.S, [self.size_train, self.size_test, self.N], axis=0)[1]
            self.test_dict[self.P['surviv']] = np.split(self.S, [self.size_train, self.size_test, self.N], axis=0)[2]

        with tf.name_scope("Survivability_Predictor"):
            self.W['surviv'] = tf.get_variable("surviv_W", shape=[dim, 1],
                                               initializer=tf.contrib.layers.xavier_initializer())
            tf.summary.histogram('weights_pred', self.W['surviv'])
            self.B['surviv'] = tf.get_variable("surviv_B", shape=[1],
                                               initializer=tf.contrib.layers.xavier_initializer())
            tf.summary.histogram('bias_pred', self.B['surviv'])
            result = black_magic(tf.add(tf.matmul(target, self.W['surviv']), self.B['surviv']), fun)
            return result

    def mirror_image(self):
        with tf.name_scope("Encoder_Optimizer_"):
            self.merged = tf.summary.merge_all()
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            with tf.Session(config=config) as sess:
                train_writer = tf.summary.FileWriter("/output/",
                                                     sess.graph)
                init = tf.global_variables_initializer()
                saver = tf.train.Saver()
                sess.run(init)
                self.spell_cast(self.item_list, sess, train_writer)
                saver.save(sess, "/tmp/model_step1.ckpt")

    def spell_cast(self, wolves, sess, tw):
        with tf.name_scope("Chain_Optimizer"):
            for wolf in wolves:
                old_train, old_vali = 0, 0
                for iter in range(wolf.epochs):
                    train_cost, _, summ = sess.run([wolf.cost, wolf.opti, self.merged], feed_dict=self.train_dict)
                    vali_cost = sess.run(wolf.cost, feed_dict=self.vali_dict)
                    if iter % (wolf.epochs / 20) == 0:
                        wolf.learn = grey_magic(wolf.learn, train_cost, old_train)
                        print("Feature: ", wolf.fea, iter, "Training Cost: ", train_cost, "Evaluation Cost: ", vali_cost)
                        tw.add_summary(summ, iter)
                    if red_magic(wolf.learn, old_train, train_cost, old_vali, vali_cost, iter, wolf.epochs) is True:
                        break
                    old_train = train_cost
                test_cost = sess.run(wolf.cost, feed_dict=self.test_dict)
                print("Feature", wolf.fea, "Test Cost: ", test_cost, "Training Cost: ", train_cost, "Evaluation Cost: ",
                      vali_cost, "Final Learning Rate: ", self.learn)

    def foresight(self, result, answer, meth, epochs, learn):
        with tf.name_scope("Survivability_Optimizer"):
            cost = tf.reduce_mean(tf.pow(result - answer, 2))
            opti = white_magic(meth, learn, cost)
            tf.summary.scalar('cost_survivability', cost)
            merged = tf.summary.merge_all()
            old_train, old_vali = 0, 0
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            with tf.Session(config=config) as sess:
                init = tf.global_variables_initializer()
                saver = tf.train.Saver()
                train_writer = tf.summary.FileWriter("/output/",
                                                     sess.graph)
                sess.run(init)
                saver.restore(sess, "/tmp/model_step1.ckpt")
                for iter in range(epochs):
                    train_cost, _, summ = sess.run([cost, opti, merged], feed_dict=self.train_dict)
                    vali_cost = sess.run(cost, feed_dict=self.vali_dict)
                    if iter % (epochs / 20) == 0:
                        print(iter, "Training Cost: ", train_cost, "Evaluation Cost: ", vali_cost)
                        learn = grey_magic(learn, train_cost, old_train)
                    if red_magic(learn, old_train, train_cost, old_vali, vali_cost, iter, epochs) is True:
                        break
                    old_train = train_cost
                    train_writer.add_summary(summ, iter)
                test_cost = sess.run(cost, feed_dict=self.test_dict)
                print("Test Cost: ", test_cost, "Training Cost: ", train_cost, "Evaluation Cost: ", vali_cost,
                      "Final Learning Rate: ", learn)

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

class FeralSpirit:
    def __init__(self, fea, result, answer, meth, epochs, learn):
        with tf.name_scope("AE_Optimizer_"+fea):
            self.fea, self.result, self.meth, self.epochs, self.learn = fea, result, meth, epochs, learn
            self.cost = tf.reduce_mean(tf.pow(result - answer, 2))
            self.opti = white_magic(meth, learn, self.cost)
            tf.summary.scalar('cost_'+fea, self.cost)
