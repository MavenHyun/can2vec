import tensorflow as tf
import numpy as np


class model:

    def __init__(self, ds):

        with tf.name_scope("Input_Layer"):
            self.cli = tf.placeholder("float", [ds.N, None])
            self.mut = tf.placeholder("float", [ds.N, None])
            self.CNV = tf.placeholder("float", [ds.N, None])
            self.mRNA = tf.placeholder("float", [ds.N, None])



