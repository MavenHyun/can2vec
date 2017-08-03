import tensorflow as tf
# NumPy is often used to load, manipulate and preprocess data.

class multi_ae:
    def __init__(self, data_set, split_aes, learn_rate):
        with tf.name_scope("Input_Layer"):
            self.L_rate = learn_rate
            self.F_cli, self.F_mut = data_set.x_cli[0].shape[1], data_set.x_mut[0].shape[1]
            self.F_CNV, self.F_mRNA = data_set.x_CNV[0].shape[1], data_set.x_mRNA[0].shape[1]
            self.F, self.SAE = self.F_cli + self.F_mut + self.F_CNV + self.F_mRNA, split_aes
            self.inputs = {'input_cli': tf.placeholder("float", [None, self.F_cli]),
                           'input_mut': tf.placeholder("float", [None, self.F_mut]),
                           'input_CNV': tf.placeholder("float", [None, self.F_CNV]),
                           'input_mRNA': tf.placeholder("float", [None, self.F_mRNA]),
                           'answer_R': tf.placeholder("float", [None, self.F]),
                           'answer_S': tf.placeholder("float", [None, 1])}
            
        with tf.name_scope("Compression_Layer"):
            self.dim_h = self.SAE[0].dim_hid + self.SAE[1].dim_hid + self.SAE[2].dim_hid + self.SAE[3].dim_hid
            self.weights = {'W_cli': tf.placeholder("float", [self.F_cli, self.SAE[0].dim_h]),
                            'W_mut': tf.placeholder("float", [self.F_mut, self.SAE[1].dim_h]),
                            'W_CNV': tf.placeholder("float", [self.F_CNV, self.SAE[2].dim_h]),
                            'W_mRNA': tf.placeholder("float", [self.F_mRNA, self.SAE[3].dim_h])}
            self.bias = {'B_cli': tf.placeholder("float", [self.SAE[0].dim_h]),
                         'B_mut': tf.placeholder("float", [self.SAE[1].dim_h]),
                         'B_CNV': tf.placeholder("float", [self.SAE[2].dim_h]),
                         'B_mRNA': tf.placeholder("float", [self.SAE[0].dim_h])}
            self.hidden = tf.concat([tf.nn.sigmoid(tf.add(tf.matmul(self.inputs['input_cli'],
                                                                    self.weights['W_cli']),
                                                          self.bias['B_cli'])),
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

        self.ingredients = {self.inputs['input_cli']: z.X_cli, self.inputs['input_mut']: z.X_mut,
                            self.inputs['input_CNV']: z.X_CNV, self.inputs['input_mRNA']: z.X_mRNA,
                            self.inputs['answer_R']: z.X, self.inputs['answer_S']: z.Y,
                            self.weights['W_cli']: s1.W, self.weights['W_mut']: s2.W,
                            self.weights['W_CNV']: s3.W, self.weights['W_mRNA']: s4.W,
                            self.bias['B_cli']: s1.B, self.bias['B_mut']: s2.B,
                            self.bias['B_CNV']: s3.B, self.bias['B_mRNA']: s4.B}

    def initiate(self):
        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init)
            for iter in range(5001):
                c, _, r = sess.run([self.cost_S, self.opti_S, self.output_S], feed_dict=self.ingredients)
                if iter % 100 == 0:
                    print(iter, "Cost is ", c)

