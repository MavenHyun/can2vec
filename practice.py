import tensorflow as tf
import numpy as np
import train_set as ts
from lifelines.utils import _naive_concordance_index
import pandas as pd


tr = ts.data_set("ACC", 45, 30)
tr.data_extract()
tr.data_preprocess()
tr.data_split()
tr.data_rearrange()


target = tf.placeholder("float", [1, None])
target2 = tf.placeholder("float", [1, None])


def cox_cummulative(target):
    target = tf.reverse(target, [-1], name='reverse_cox')
    #target = tf.exp(target, name='exp_cox')
    target = tf.slice(target, [0, 0], [-1, 45], name='slice_cox')
    values = tf.split(target, target.get_shape()[1], 1, name='split_cox')
    csum = tf.zeros_like(values[0], name='zeros_cox')
    out = []
    for val in values:
        out.append(csum)
        csum = tf.add(csum, val, name='add_cox')
    result = tf.reverse(tf.concat(out, 1, name='concat_cox'), [-1], name='unverse_cox')
    return result

sth = cox_cummulative(target)

sess = tf.Session()
t = sess.run(sth, feed_dict={target: tr.T['sur']})
print(t)
