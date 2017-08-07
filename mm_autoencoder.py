import tensorflow as tf
import numpy as np
import train_set as ts


class model:

    def __init__(self, ):
        x=0


with tf.name_scope("Input_Layer"):
    cli = tf.placeholder("float", [92, None])
    mut = tf.placeholder("float", [92, None])
    CNV = tf.placeholder("float", [92, None])
    mRNA = tf.placeholder("float", [92, None])

with tf.name_scope("Weights and Bias"):
    w = {
        'w_cli_e': tf.Variable(tf.random_normal([None, 10])),
        'w_cli_d': tf.Variable(tf.random_normal([10, None])),
        'w_mut_e': tf.Variable(tf.random_normal([None, 40])),
        'w_mut_d': tf.Variable(tf.random_normal([40, None])),
        'w_CNV1_e': tf.Variable(tf.random_normal([None, 1000])),
        'w_CNV2_e': tf.Variable(tf.random_normal([1000, 500])),
        'w_mRNA1_e': tf.Variable(tf.random_normal([None, 8000])),
        'w_mRNA2_e': tf.Variable(tf.random_normal([8000, 8000])),

    }

    b = {
        'w_cli_e': tf.Variable(tf.random_normal([None, 10])),
        'w_cli_d': tf.Variable(tf.random_normal([10, None])),
        'w_mut_e': tf.Variable(tf.random_normal([None, 40])),
        'w_mut_d': tf.Variable(tf.random_normal([40, None])),
        'w_CNV1_e': tf.Variable(tf.random_normal([None, 1000])),
        'w_CNV2_e': tf.Variable(tf.random_normal([1000, 500])),
        'w_CNV2_d': tf.Variable(tf.random_normal([500, 1000])),
        'w_CNV1_d': tf.Variable(tf.random_normal([1000, None])),
        'w_mRNA1_e': tf.Variable(tf.random_normal([None, 12000])),
        'w_mRNA2_e': tf.Variable(tf.random_normal([12000, 8000])),
        'w_mRNA3_e': tf.Variable(tf.random_normal([8000, 4450])),
        'w_mRNA3_d': tf.Variable(tf.random_normal([4450, 8000])),
        'w_mRNA2_d': tf.Variable(tf.random_normal([8000, 12000])),
        'w_mRNA1_d': tf.Variable(tf.random_normal([12000, None]))
    }

def encoded(target, name, dim0, dim1, act_funct):
    with tf.name_scope(name):
        w = tf.Variable(tf.random_normal([dim0, dim1]))
        b = tf.Variable(tf.random_normal([dim1]))
        if act_funct == 0:
            result = tf.nn.relu(tf.add(tf.matmul(target, w), b))
        elif act_funct == 1:
            result = tf.nn.sigmoid(tf.add(tf.matmul(target, w), b))
        elif act_funct == 2:
            result = tf.nn.tanh(tf.add(tf.matmul(target, w), b))
        else:
            result = tf.add(tf.matmul(target, w), b)
    return result

def decoded(target, name, dim0, dim1, act_funct):
    with tf.name_scope(name):
        w = tf.Variable(tf.random_normal([dim0, dim1]))
        b = tf.Variable(tf.random_normal([dim1]))
        if act_funct == 0:
            result = tf.nn.relu(tf.add(tf.matmul(target, w), b))
        elif act_funct == 1:
            result = tf.nn.sigmoid(tf.add(tf.matmul(target, w), b))
        elif act_funct == 2:
            result = tf.nn.tanh(tf.add(tf.matmul(target, w), b))
        else:
            result = tf.add(tf.matmul(target, w), b)
    return result






