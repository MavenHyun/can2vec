import tensorflow as tf
# NumPy is often used to load, manipulate and preprocess data.

class multi_ae:
    def __init__(self, z, s1, s2, s3, s4, lr):

        with tf.name_scope("Input_Layer"):
            self.learning_rate = lr
            self.inputs = {'input_categ': tf.placeholder("float", [z.X_categ.shape[0], z.X_categ.shape[1]]),
                           'input_mut': tf.placeholder("float", [z.X_mut.shape[0], z.X_mut.shape[1]]),
                           'input_CNV': tf.placeholder("float", [z.X_CNV.shape[0], z.X_CNV.shape[1]]),
                           'input_mRNA': tf.placeholder("float", [z.X_mRNA.shape[0], z.X_mRNA.shape[1]]),
                           'answer_R': tf.placeholder("float", [z.X.shape[0], z.X.shape[1]]),
                           'answer_S': tf.placeholder("float", [z.Y.shape[0], z.Y.shape[1]])}
            
        with tf.name_scope("Compression_Layer"):
            self.dim_h = s1.dim_h + s2.dim_h + s3.dim_h + s4.dim_h
            self.weights = {'W_categ': tf.placeholder("float", [z.X_categ.shape[1], s1.dim_h]),
                            'W_mut': tf.placeholder("float", [z.X_mut.shape[1], s2.dim_h]),
                            'W_CNV': tf.placeholder("float", [z.X_CNV.shape[1], s3.dim_h]),
                            'W_mRNA': tf.placeholder("float", [z.X_mRNA.shape[1], s4.dim_h])}
            self.bias = {'B_categ': tf.placeholder("float", [s1.dim_h]),
                         'B_mut': tf.placeholder("float", [s2.dim_h]),
                         'B_CNV': tf.placeholder("float", [s3.dim_h]),
                         'B_mRNA': tf.placeholder("float", [s4.dim_h])}
            self.hidden = tf.concat([tf.nn.sigmoid(tf.add(tf.matmul(self.inputs['input_categ'],
                                                                    self.weights['W_categ']),
                                                          self.bias['B_categ'])),
                                     tf.nn.sigmoid(tf.add(tf.matmul(self.inputs['input_mut'],
                                                                    self.weights['W_mut']),
                                                          self.bias['B_mut'])),
                                     tf.nn.sigmoid(tf.add(tf.matmul(self.inputs['input_CNV'],
                                                                    self.weights['W_CNV']),
                                                          self.bias['B_CNV'])),
                                     tf.nn.sigmoid(tf.add(tf.matmul(self.inputs['input_mRNA'],
                                                                    self.weights['W_mRNA']),
                                                          self.bias['B_mRNA']))], 1)
            
        with tf.name_scope("Alchemy_Layer"):
            self.weight_A = tf.Variable(tf.random_normal([self.dim_h, self.dim_h]))
            self.bias_A = tf.Variable(tf.random_normal([self.dim_h]))
            self.alchemy = tf.nn.sigmoid(tf.add(tf.matmul(self.hidden, self.weight_A), self.bias_A))
            
        with tf.name_scope("Survivability_Layer"):
            self.weight_S = tf.Variable(tf.random_normal([self.dim_h, z.Y.shape[1]]))
            self.bias_S = tf.Variable(tf.random_normal([z.Y.shape[1]]))
            self.output_S = tf.nn.sigmoid(tf.add(tf.matmul(self.alchemy, self.weight_S), self.bias_S))

        with tf.name_scope("Reconstruction_Layer"):
            self.weight_R = tf.Variable(tf.random_normal([self.dim_h, z.X.shape[1]]))
            self.bias_R = tf.Variable(tf.random_normal([z.X.shape[1]]))
            self.output_R = tf.nn.sigmoid(tf.add(tf.matmul(self.hidden, self.weight_R), self.bias_R))

        with tf.name_scope("Optimus_Prime"):
            self.cost_S = tf.reduce_mean(tf.pow(self.inputs['answer_S'] - self.output_S, 2))
            self.opti_S = tf.train.AdagradOptimizer(self.learning_rate).minimize(self.cost_S)
            self.cost_R = tf.reduce_mean(tf.pow(self.inputs['answer_R'] - self.output_R, 2))
            self.opti_R = tf.train.AdagradOptimizer(self.learning_rate).minimize(self.cost_R)

        self.ingredients = {self.inputs['input_categ']: z.X_categ, self.inputs['input_mut']: z.X_mut,
                            self.inputs['input_CNV']: z.X_CNV, self.inputs['input_mRNA']: z.X_mRNA,
                            self.inputs['answer_R']: z.X, self.inputs['answer_S']: z.Y,
                            self.weights['W_categ']: s1.W, self.weights['W_mut']: s2.W,
                            self.weights['W_CNV']: s3.W, self.weights['W_mRNA']: s4.W,
                            self.bias['B_categ']: s1.B, self.bias['B_mut']: s2.B,
                            self.bias['B_CNV']: s3.B, self.bias['B_mRNA']: s4.B}

    def initiate(self):

        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init)
            for iter in range(5001):
                c, _, r = sess.run([self.cost_S, self.opti_S, self.output_S], feed_dict=self.ingredients)
                if iter % 100 == 0:
                    print(iter, "Cost is ", c)
                    print(r)
