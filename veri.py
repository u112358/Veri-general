from __future__ import absolute_import, division, print_function

import tensorflow as tf
import numpy as np
import scipy.io as sio
import Deep_Net as gan
import ReadBatches as data_reader
from six.moves import xrange


batch_size = 20
address="../../dataset/ZSL/APY/"
learning_rate = 0.0001

max_iter = 400000


def main(argv=None):

    reader = data_reader.BatchDatset(address, batch_size)
    feat_input = tf.placeholder(tf.float32, [batch_size, reader.feat_len+reader.attr_len])
    phase_train = tf.placeholder(tf.bool)
    sim_input = tf.placeholder(tf.float32, [batch_size, 1])

    # generate feature and discriminate
    logits = gan.Classify_Net(feat_input, phase_train=phase_train)
    # logits = tf.reduce_mean(tf.square(logits), axis=1, keep_dims=True)

    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=sim_input))
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

    for T in xrange(max_iter):
        # read features and labels
        feat = reader.next_batch_train()

        _, loss_o = sess.run([opt, loss], feed_dict={feat_input: feat[:, :reader.feat_len+reader.attr_len],\
                                                     sim_input: feat[:, reader.feat_len+reader.attr_len:], \
                                                     phase_train:True})

        if T % 200 == 0:
            print("Iteration " + str(T) + ": " + "Loss=" + str(loss_o))
        # if T % 20000 == 1000:
            # saver.save(sess, 'models/graph.ckpt')
    # try to generate fake feature samples for zero-shot training
        if (T+1)%2000==0:
            test_size = np.shape(reader.test_pairs)[0]
            out_loss = np.zeros([test_size, 1], dtype=np.float32)
            out_real = np.zeros([test_size, 1], dtype=np.float32)
            for i in xrange(int(test_size/batch_size)):
                feat = reader.next_batch_test()
                tmp_out = sess.run( gen_out, feed_dict={feat_input: feat[:, :reader.feat_len+reader.attr_len], phase_train:False})
                out_loss[i * batch_size:(i + 1) * batch_size, :] = tmp_out
                out_real[i * batch_size:(i + 1) * batch_size, :] = feat[:, reader.feat_len+reader.attr_len:]
            count = 0
            test_num = np.shape(reader.test_feat)[0]
            num_test_cls = np.shape(reader.testClass)[0]
            for i in xrange(test_num):
                tmp_out = out_loss[i*num_test_cls: (i+1)*num_test_cls]
                tmp_real = out_real[i*num_test_cls: (i+1)*num_test_cls]
                pos = np.argmin(tmp_out, axis=0)
                if tmp_real[pos]<=0.9:
                    count += 1
            acc = count/test_num
            print("acc="+ str(acc))

if __name__ == "__main__":
    tf.app.run()
