import tensorflow as tf
import tf_components as tc
import split_ae as sa
# NumPy is often used to load, manipulate and preprocess data.

class multi_ae:
    def __init__(self, data_set, products_cli, products_mut, products_CNV, products_mRNA, total_hnodes):

        with tf.name_scope("Basic_Settings"):
            self.H, self.F = total_hnodes, data_set.X.shape[1]

        with tf.name_scope("Placeholders"):
            self.inputs = {'encoded_cli': tf.placeholder("float", [None, None]),
                           'encoded_mut': tf.placeholder("float", [None, None]),
                           'encoded_CNV': tf.placeholder("float", [None, None]),
                           'encoded_mRNA': tf.placeholder("float", [None, None]),
                           'answer_R': tf.placeholder("float", [None, self.H]),
                           'answer_S': tf.placeholder("float", [None, 1])}
            self.feature_vector = tf.concat([self.inputs['encoded_cli'], self.inputs['encoded_mut'],
                                             self.inputs['encoded_CNV'], self.inputs['encoded_mRNA']], 1)

        with tf.name_scope("Dictionaries"):
            self.train_dict = {self.inputs['encoded_cli']: products_cli[0],
                               self.inputs['encoded_mut']: products_mut[0],
                               self.inputs['encoded_CNV']: products_CNV[0],
                               self.inputs['encoded_mRNA']: products_mRNA[0],
                               self.inputs['answer_S']: data_set.y[0],
                               self.inputs['answer_R']: data_set.x[0]}
            self.eval_dict = {self.inputs['encoded_cli']: products_cli[1],
                              self.inputs['encoded_mut']: products_mut[1],
                              self.inputs['encoded_CNV']: products_CNV[1],
                              self.inputs['encoded_mRNA']: products_mRNA[1],
                              self.inputs['answer_S']: data_set.y[1],
                              self.inputs['answer_R']: data_set.x[1]}
            self.test_dict = {self.inputs['encoded_cli']: products_cli[2],
                              self.inputs['encoded_mut']: products_mut[2],
                              self.inputs['encoded_CNV']: products_CNV[2],
                              self.inputs['encoded_mRNA']: products_mRNA[2],
                              self.inputs['answer_S']: data_set.y[2],
                              self.inputs['answer_R']: data_set.x[2]}

    def construct_Projector(self, layer_name, activation):
        pro = tc.Projector(layer_name, self.feature_vector, self.H, activation)
        return pro

    def construct_Predictor(self, layer_name, input_ph, activation):
        pre = tc.Predictor(layer_name, self.H, input_ph, activation)
        return pre

    def construct_ReConstructor(self, layer_name, activation):
        rec = tc.ReConstructor(layer_name, self.feature_vector, self.H, self.F, activation)
        return rec

    def construct_Optimizer(self, layer_name, output_ph, answer_ph, learn_rate, train_meth):
        opt = tc.Optimizer(layer_name, output_ph, answer_ph, learn_rate, train_meth)
        return opt

    def initiate(self, optimizer, epochs):
        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init)
            optimizer.optimize_ma(sess, self.train_dict, self.eval_dict, self.test_dict, epochs)


