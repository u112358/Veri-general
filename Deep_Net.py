from __future__ import absolute_import, division, print_function

import tensorflow as tf
import utils


def Classify_Net(feat, attr, reuse=False):
    with tf.variable_scope("classify", reuse=reuse):
        # fc2
        len1 = feat.get_shape().as_list()[1]
        len2 = attr.get_shape().as_list()[1]
        kernels = utils.weight_variable(shape=[len1+len2, 2048], name="fc1_w")
        bias = utils.weight_variable([2048], name="fc1_b")
        current = tf.nn.xw_plus_b(tf.concat([feat,attr],axis=1), kernels, bias)
        current = tf.nn.relu(current)

        kernels = utils.weight_variable(shape=[2048, 1024], name="fc2_w")
        bias = utils.weight_variable([1024], name="fc2_b")
        current = tf.nn.xw_plus_b(current, kernels, bias)
        current = tf.nn.relu(current)

        kernels = utils.weight_variable(shape=[1024, 512], name="fc4_w")
        bias = utils.weight_variable([512], name="fc4_b")
        current = tf.nn.xw_plus_b(current, kernels, bias)
        current = tf.nn.relu(current)

        kernels = utils.weight_variable(shape=[512, 1], name="fc3_w")
        bias = utils.weight_variable([1], name="fc3_b")
        fc2 = tf.nn.xw_plus_b(current, kernels, bias)

        # product = tf.reduce_sum(tf.multiply(fc2, attr), axis=1, keep_dims=True)
        #
        # kernels = utils.weight_variable(shape=[len2, 2048], name="fc4_w")
        # bias = utils.weight_variable([2048], name="fc4_b")
        # current = tf.nn.xw_plus_b(fc2, kernels, bias)
        # current = tf.nn.relu(current)
        #
        # kernels = utils.weight_variable(shape=[2048, 2048], name="fc5_w")
        # bias = utils.weight_variable([2048], name="fc5_b")
        # fc2 = tf.nn.xw_plus_b(current, kernels, bias)

    return  fc2

def Classify_Net_Product(feat, attr, reuse=False):
    with tf.variable_scope("classify", reuse=reuse):
        # fc2
        len1 = feat.get_shape().as_list()[1]
        len2 = attr.get_shape().as_list()[1]
        W1 = utils.weight_variable(shape=[len1, 2048], name="fc1_w")
        bias = utils.weight_variable([2048], name="fc1_b")
        current = tf.nn.xw_plus_b(feat, W1, bias)
        current = tf.nn.relu(current)
        len3 = int(len2*5)
        W2 = utils.weight_variable(shape=[2048, len3], name="fc2_w")
        bias = utils.weight_variable([len3], name="fc2_b")
        fc2 = tf.nn.xw_plus_b(current, W2, bias)
        # fc2 = tf.nn.relu(fc2)

        W3 = utils.weight_variable(shape=[len2, len3], name="fc3_w")
        bias = utils.weight_variable([len3], name="fc3_b")
        fc3 = tf.nn.xw_plus_b(attr, W3, bias)
        # fc3 = tf.nn.relu(fc3)

        product = tf.reduce_sum(tf.multiply(fc2, fc3), axis=1, keep_dims=True)

        # W2 = utils.weight_variable(shape=[len3, len3], name="fc4_w")
        bias = utils.weight_variable([len2], name="fc4_b")
        fc2 = tf.nn.xw_plus_b(fc2, tf.transpose(W3), bias)
        fc2 = tf.nn.relu(fc2)

        # kernels = utils.weight_variable(shape=[len2*10, 2048], name="fc4_w")
        bias = utils.weight_variable([2048], name="fc5_b")
        current = tf.nn.xw_plus_b(fc3, tf.transpose(W2), bias)
        current = tf.nn.relu(current)

        # kernels = utils.weight_variable(shape=[len2*10, len2], name="fc6_w")
        bias = utils.weight_variable([len1], name="fc6_b")
        fc3 = tf.nn.xw_plus_b(current, tf.transpose(W1), bias)
        fc3 = tf.nn.relu(fc3)

    return  fc2, fc3, product

def Naive_Net(feat,attr,reuse=False):
    with tf.variable_scope("naive",reuse = reuse):
        len1 = feat.get_shape().as_list()[1]
        len2 = attr.get_shape().as_list()[1]
        W1 = utils.weight_variable(shape=[len1, 2048], name="fc1_w")
        bias = utils.weight_variable([2048], name="fc1_b")
        current = tf.nn.xw_plus_b(feat, W1, bias)
        current = tf.nn.relu(current)
        len3 = int(len2 * 5)
        W2 = utils.weight_variable(shape=[2048, len3], name="fc2_w")
        bias = utils.weight_variable([len3], name="fc2_b")
        fc2 = tf.nn.xw_plus_b(current, W2, bias)
        fc2 = tf.nn.l2_normalize(fc2,dim=1,epsilon=1e-12,name="fc2_normalized")
        # fc2 = tf.nn.relu(fc2)

        W3 = utils.weight_variable(shape=[len2, len3], name="fc3_w")
        bias = utils.weight_variable([len3], name="fc3_b")
        fc3 = tf.nn.xw_plus_b(attr, W3, bias)
        fc3 = tf.nn.l2_normalize(fc3,dim=1,epsilon=1e-12,name="fc3_normalized")

        dis = tf.reduce_sum(tf.square(tf.subtract(fc2,fc3)),1)
    return fc2,fc3,dis