from __future__ import absolute_import, division, print_function

import tensorflow as tf
import numpy as np
import scipy.io as sio
import Deep_Net as gan
import ReadBatches as data_reader
from six.moves import xrange
import os
from datetime import datetime
import argparse
import sys

batch_size = 20
address = "./"

max_iter = 2000000
threshold = 1.06


def main(args):
    margin = args.margin
    w_loss1 = args.w_loss1
    w_loss2 = args.w_loss2
    learning_rate = args.lr
    reader = data_reader.BatchDatset(address, batch_size)
    feat_input = tf.placeholder(tf.float32, [None, reader.feat_len])
    attr_input = tf.placeholder(tf.float32, [None, reader.attr_len])
    sim_input = tf.placeholder(tf.float32, [None, 1])
    # generate feature and discriminate
    # fc2, fc3, logits = gan.Classify_Net_Product(feat_input, attr_input)
    #
    # loss1 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=sim_input))
    # # loss2 = tf.reduce_mean(tf.square(fc2-feat_input))+ tf.reduce_mean(tf.square(fc3-attr_input))
    # loss2 = tf.reduce_mean(tf.reduce_mean(tf.square(fc3 - feat_input), axis=1) * sim_input)
    # loss3 = tf.reduce_mean(tf.reduce_mean(tf.square(fc2 - attr_input), axis=1) * sim_input)
    # # loss = loss1 + 1000*loss2 + 0.001*loss3
    # loss = loss1



    fc2, fc3, gen_out = gan.Naive_Net(feat_input, attr_input)
    d_sqrt = tf.sqrt(gen_out)
    loss1 = tf.reduce_mean((1.0 - sim_input) * tf.square(tf.maximum(0., margin - d_sqrt)))
    loss2 = tf.reduce_mean(sim_input * gen_out)
    loss = w_loss1 * loss1 + w_loss2 * loss2

    training_list = tf.trainable_variables()
    opt = tf.train.AdamOptimizer(learning_rate).minimize(loss, var_list=training_list)
    # gen_out = tf.nn.sigmoid(logits)

    sess = tf.Session()
    init_op = tf.global_variables_initializer()
    sess.run(init_op)

    # subdir = datetime.strftime(datetime.now(), '%Y%m%d-%H%M%S')
    subdir = 'margin_%.2f_w1_%.2f_w2_%.2f_lr_%f' % (margin, w_loss1, w_loss2, learning_rate)
    summary_writer = tf.summary.FileWriter(os.path.join('./log_on_contrastive/', subdir), sess.graph)
    # ckpt = tf.train.get_checkpoint_state('models')
    # if ckpt and ckpt.model_checkpoint_path:
    #     saver.restore(sess, ckpt.model_checkpoint_path)
    #     print('...............Load from pre-trained graph!........................')

    print("......................Start to training deep models...................")
    for T in xrange(max_iter):
        # read features and labels
        summary = tf.Summary()
        feat = reader.next_batch_train()
        _, loss_o = sess.run([opt, loss], feed_dict={feat_input: feat[:, :reader.feat_len],
                                                     attr_input: feat[:,
                                                                 reader.feat_len:reader.feat_len + reader.attr_len],
                                                     sim_input: feat[:, reader.feat_len + reader.attr_len:]})
        if T % 200 == 0:
            print("Iteration " + str(T) + ": " + "Loss=" + str(loss_o))
            # if T % 20000 == 1000:
            # saver.save(sess, 'models/graph.ckpt')
            summary.value.add(tag='loss', simple_value=loss_o)
        # try to generate fake feature samples for zero-shot training
        if (T + 1) % 1000 == 0:
            # for tr accuracy
            # print('only test')
            # num_att = np.shape(reader.class_attributes)[0]
            # num_train = np.shape(reader.train_label)[0]
            # res_array=[]
            # lab_array=[]
            # count = 0
            # for i in xrange(num_train):
            #     feature, lab = reader.next_feat_train(i)
            #     feature = np.tile(feature, [num_att, 1])
            #     att = reader.class_attributes
            #     res_out = sess.run(gen_out, feed_dict={feat_input: feature, attr_input: att})
            #     res_array.append(res_out)
            #     lab_array.append(lab)
            #     pos = np.argmax(res_out, axis=0)
            #     if lab - 1 == pos:
            #         count += 1
            #
            # filename = 'result_tr%d.mat' % T
            # sio.savemat(filename,{'res':res_array,'lab':lab_array})
            # acc_tr = count / num_train
            # acc[times, 0] = acc_tr
            # print("acc_tr=" + str(acc_tr))

            # for ts accuracy
            att = reader.class_attributes[reader.testClass - 1]
            num_att = np.shape(att)[0]
            # num_att = np.shape(reader.class_attributes)[0]
            num_test = np.shape(reader.test_label)[0]
            res_array = []
            lab_array = []
            count = 0
            for i in xrange(num_test):
                feature, lab = reader.next_feat_test(i)
                feature = np.tile(feature, [num_att, 1])
                # att = reader.class_attributes
                res_out = sess.run(gen_out, feed_dict={feat_input: feature, attr_input: att})
                res_array.append(res_out)
                lab_array.append(lab)
                pos = np.argmin(res_out, axis=0)
                sorted = np.sort(res_out, axis=0)[::-1]
                ratio = sorted[0] / sorted[1]
                if lab == reader.testClass[pos]:
                    count += 1

                    # if ratio>threshold:
                    #     sim_label = np.zeros([num_att,1])
                    #     sim_label[pos]=1
                    #     sess.run([opt, loss], feed_dict={feat_input: feature,
                    #                                      attr_input: att,
                    #                                      sim_input: sim_label})

            acc_ts = count / num_test
            print("acc_ts=" + str(acc_ts))
            summary.value.add(tag='acc_ts', simple_value=acc_ts)
        summary_writer.add_summary(summary, global_step=T)
        summary_writer.flush()


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--margin', type=float,
                        help='margin', default=1)
    parser.add_argument('--w_loss1', type=float,
                        help='weight of loss1', default=0.5)
    parser.add_argument('--w_loss2', type=float,
                        help='weight of loss2', default=0.5)
    parser.add_argument('--lr', type=float,
                        help='learning rate', default=0.0001)
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_arguments(sys.argv[1:])
    main(args)
