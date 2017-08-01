import tensorflow as tf

class split_ae:

    def __init__(self, x, dh, lr):

        with tf.name_scope("Input_Layer"):
            self.X, self.dim_h, self.learning_rate = x, dh, lr
            self.input_data = tf.placeholder("float", [self.X.shape[0], self.X.shape[1]])

        with tf.name_scope("Encoding_Layer"):
            self.W_enc = tf.Variable(tf.random_normal([self.X.shape[1], self.dim_h]))
            self.B_enc = tf.Variable(tf.random_normal([self.dim_h]))
            self.enc = tf.nn.sigmoid(tf.add(tf.matmul(self.input_data, self.W_enc), self.B_enc))

        with tf.name_scope("Decoding_Layer"):
            self.W_dec = tf.Variable(tf.random_normal([self.dim_h, self.X.shape[1]]))
            self.B_dec = tf.Variable(tf.random_normal([self.X.shape[1]]))
            self.dec = tf.nn.sigmoid(tf.add(tf.matmul(self.enc, self.W_dec), self.B_dec))

        with tf.name_scope("Optimus_Prime"):
            self.C = tf.reduce_mean(tf.pow(self.input_data - self.dec, 2))
            self.O = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.C)

    def initiate(self):
        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init)
            for iter in range(1001):
                _, _, self.W, self.B = sess.run([self.C, self.O, self.W_enc, self.B_enc],
                                                feed_dict={self.input_data: self.X})
                if iter % 100 == 0:
                    print(iter)
            print("Optimization done!")