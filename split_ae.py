import tensorflow as tf

class split_ae:

    def __init__(self, x, y, z, dh, lr):

        with tf.name_scope("Input_Layer"):
            self.X, self.Y, self.Z, self.dim_h, self.learning_rate = x, y, z, dh, lr
            self.input_data = tf.placeholder("float", [None, self.X.shape[1]])

        with tf.name_scope("Encoding_Layer"):
            self.W_enc = tf.Variable(tf.random_normal([self.X.shape[1], self.dim_h]))
            self.B_enc = tf.Variable(tf.random_normal([self.dim_h]))
            self.enc = tf.nn.tanh(tf.add(tf.matmul(self.input_data, self.W_enc), self.B_enc))

        with tf.name_scope("Decoding_Layer"):
            self.W_dec = tf.Variable(tf.random_normal([self.dim_h, self.X.shape[1]]))
            self.B_dec = tf.Variable(tf.random_normal([self.X.shape[1]]))
            self.dec = tf.nn.tanh(tf.add(tf.matmul(self.enc, self.W_dec), self.B_dec))

        with tf.name_scope("Optimus_Prime"):
            self.C = tf.reduce_mean(tf.pow(self.input_data - self.dec, 2))
            self.O = tf.train.AdamOptimizer(self.learning_rate).minimize(self.C)

    def initiate(self):
        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init)
            writer = tf.summary.FileWriter("C:/Users/Arthur Keonwoo Kim/PycharmProjects/can2vec", sess.graph)
            for iter in range(10001):
                train_cost, _, self.W, self.B = sess.run([self.C, self.O, self.W_enc, self.B_enc],
                                                feed_dict={self.input_data: self.X})
                if iter % 100 == 0:
                    print(iter, "th step:\t", "Cost is ", train_cost)
                    eval_cost = sess.run(self.C, feed_dict={self.input_data: self.Y})
                    print("Evaluation: ", eval_cost)

            eval_cost = sess.run(self.C, feed_dict={self.input_data: self.Z})
            print("Test: ", eval_cost)
