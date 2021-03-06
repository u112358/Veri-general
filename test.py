from __future__ import absolute_import, division, print_function

import tensorflow as tf
import numpy as np
import scipy.io as sio
import Deep_Net as gan
import ReadBatches as data_reader
from six.moves import xrange


batch_size = 20
address="./"

learning_rate = 0.0001

max_iter = 100000


def main(argv=None):

    reader = data_reader.BatchDatset(address, batch_size)
    feat_input = tf.placeholder(tf.float32, [None, reader.feat_len])
    attr_input = tf.placeholder(tf.float32, [None, reader.attr_len])
    sim_input = tf.placeholder(tf.float32, [None, 1])

    # generate feature and discriminate
    fc2, fc3, logits = gan.Classify_Net_Product(feat_input, attr_input)

    loss1 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=sim_input))
    # loss2 = tf.reduce_mean(tf.square(fc2-feat_input))+ tf.reduce_mean(tf.square(fc3-attr_input))
    loss2 = tf.reduce_mean(tf.reduce_mean(tf.square(fc3-feat_input), axis=1)*sim_input)
    loss3 = tf.reduce_mean(tf.reduce_mean(tf.square(fc2-attr_input), axis=1)*sim_input)
    # loss = loss1 + 1000*loss2 + 0.001*loss3
    loss = loss1

    training_list = tf.trainable_variables()
    opt = tf.train.AdamOptimizer(learning_rate).minimize(loss, var_list=training_list)
    gen_out = tf.nn.sigmoid(logits)

    sess = tf.Session()
    init_op = tf.global_variables_initializer()
    sess.run(init_op)

    # ckpt = tf.train.get_checkpoint_state('models')
    # if ckpt and ckpt.model_checkpoint_path:
    #     saver.restore(sess, ckpt.model_checkpoint_path)
    #     print('...............Load from pre-trained graph!........................')

    print("......................Start to training deep models...................")
    times = 0
    acc = np.zeros([200, 3], dtype=np.float32)
    for T in xrange(max_iter):
        # read features and labels
        feat = reader.next_batch_train()
        _, loss_o = sess.run([opt, loss], feed_dict={feat_input: feat[:, :reader.feat_len], \
                                                     attr_input: feat[:, reader.feat_len:reader.feat_len + reader.attr_len], \
                                                     sim_input: feat[:, reader.feat_len+reader.attr_len:]})
        if T % 200 == 0:
            print("Iteration " + str(T) + ": " + "Loss=" + str(loss_o))
        # if T % 20000 == 1000:
            # saver.save(sess, 'models/graph.ckpt')
        # try to generate fake feature samples for zero-shot training
        if (T+1)%3000==0:
            # for tr accuracy
            num_att = np.shape(reader.class_attributes)[0]
            num_train = np.shape(reader.train_label)[0]
            count = 0
            for i in xrange(num_train):
                feature, lab = reader.next_feat_train(i)
                feature = np.tile(feature,[num_att,1])
                att = reader.class_attributes
                res_out = sess.run(gen_out, feed_dict={feat_input: feature, attr_input: att})
                pos = np.argmax(res_out, axis=0)
                if lab-1 == pos:
                    count += 1
            acc_tr = count/num_train
            acc[times, 0] = acc_tr
            print("acc_tr="+ str(acc_tr))
            # for ts accuracy
            num_att = np.shape(reader.class_attributes)[0]
            num_test = np.shape(reader.test_label)[0]
            count = 0
            for i in xrange(num_test):
                feature, lab = reader.next_feat_test(i)
                feature = np.tile(feature, [num_att, 1])
                att = reader.class_attributes
                res_out = sess.run(gen_out, feed_dict={feat_input: feature, attr_input: att})
                pos = np.argmax(res_out, axis=0)
                if lab - 1 == pos:
                    count += 1
            acc_ts = count / num_test

            acc[times, 1] = acc_ts
            print("acc_ts=" + str(acc_ts))
            acc_H = 2*acc_tr*acc_ts/(acc_tr+acc_ts)
            acc[times, 2] = acc_H
            print("acc_H=" + str(acc_H))
            times += 1
    sio.savemat('awa_tstrH.mat', mdict={'acc':acc})

if __name__ == "__main__":
    tf.app.run()
