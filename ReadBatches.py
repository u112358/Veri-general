"""
Code ideas from https://github.com/Newmu/dcgan and tensorflow mnist dataset reader
"""
import numpy as np
import scipy.io as sio
import os.path as op
from six.moves import xrange

class BatchDatset:

    def __init__(self, data_dir, batch_size):
        """
        data_dir is the directory of the input data, which should be the format of 'XX/XX/XX/'
        total_batches is the number of the batches, which begins from 1.mat to total_batches.mat
        """
        print("Initializing Batch Dataset Reader...")
        self.batch_size = batch_size
        np.random.seed(11235)
        self.rand_size = 20
        self.epoch = 0

        self.data = sio.loadmat(data_dir+'sun.mat')
        self.class_attributes = self.data['att']
        self.cnn_feat = self.data['features']
        self.labels = np.squeeze(self.data['labels'], axis=1)
        self.label_cv = self.class_attributes[self.labels-1, :]
        self.testClass = np.squeeze(self.data['test_class'], axis=1)
    #     process the sub-dataset for training
        test_ind = []
        total_len = np.shape(self.labels)[0]
        len = np.shape(self.testClass)[0]
        for i in xrange(len):
            k = self.testClass[i]
            arr = np.where(self.labels==k)[0]
            test_ind = np.concatenate((test_ind, arr))
        test_ind = test_ind.astype(np.int32)
        self.test_feat =self.cnn_feat[test_ind, :]
        self.test_label_cv = self.label_cv[test_ind, :]
        self.test_label = self.labels[test_ind]
        train_ind = np.delete(np.arange(total_len), test_ind, axis=0)
        self.train_feat = self.cnn_feat[train_ind, :]
        self.train_label_cv = self.label_cv[train_ind, :]
        self.train_label = self.labels[train_ind]
        self.feat_len = np.shape(self.test_feat)[1]
        self.attr_len = np.shape(self.test_label_cv)[1]

        #############################################
        #construct similarity and dissimilarity batches for training
        self.train_batch_id = 0
        self.test_batch_id = 0
        self.train_pairs = self.train_batch_generation()
        self.train_total_batches = int(np.shape(self.train_pairs)[0]/self.batch_size)


    def train_batch_generation(self):
        num_train = np.shape(self.train_label)[0]
        len_feat = np.shape(self.train_feat)[1]
        len_attr = np.shape(self.train_label_cv)[1]
        sim_pairs = np.zeros([num_train, len_feat+len_attr+1], dtype=np.float32)
        dsim_pairs = np.zeros([num_train, len_feat+len_attr+1], dtype=np.float32)
        ulabel = np.unique(self.train_label)
        train_len = np.shape(ulabel)[0]
        end = 0
        for i in xrange(train_len):
            ind1 = np.where(self.train_label == ulabel[i])[0]
            np.random.shuffle(ind1)
            ind2 = np.copy(ind1)
            np.random.shuffle(ind2)
            rand_size = np.shape(ind1)[0]
            sim = np.ones([rand_size, 1], dtype=np.float32)
            vector = np.concatenate([self.train_feat[ind1[0:rand_size], :], \
                                     self.train_label_cv[ind2[0:rand_size], :], sim], axis=1)
            start = end
            end = start + rand_size
            sim_pairs[start:end, :] = vector

            if i %100 == 0:
                print("generating simliar train batches"+ str(i))
        num_train = np.shape(self.train_label)[0]
        end = 0
        for i in xrange(train_len):
            ind1 = np.where(self.train_label == ulabel[i])[0]
            np.random.shuffle(ind1)
            ind2 = np.delete(np.arange(num_train), ind1, axis=0)
            np.random.shuffle(ind2)
            rand_size = np.shape(ind1)[0]
            sim = np.zeros([rand_size, 1], dtype=np.float32)
            vector = np.concatenate([self.train_feat[ind1[0:rand_size], :], \
                                     self.train_label_cv[ind2[0:rand_size], :], sim], axis=1)
            start = end
            end = start + np.shape(ind1)[0]
            dsim_pairs[start:end, :] = vector

            if i %50 == 0:
                print("generating dissimilar train batches"+ str(i))
        pairs = np.concatenate([sim_pairs, dsim_pairs], axis=0)
        ind = np.arange(np.shape(pairs)[0])
        np.random.shuffle(ind)
        pairs = pairs[ind, :]
        
        return pairs


    def next_batch_train(self):
        if self.train_batch_id >= self.train_total_batches:
            self.train_pairs = self.train_batch_generation()
            self.train_batch_id = 0
            string = op.join('------------ finished', str(self.epoch))
            string = op.join(string, 'epoch---------------')
            print(string)
            self.epoch += 1
        start = self.train_batch_id * self.batch_size
        end = (self.train_batch_id + 1) * self.batch_size
        # label_id = self.train_label[start:end].astype(np.int32)
        self.train_batch_id += 1
        return self.train_pairs[start:end, :].astype(np.float32)


    def next_feat_train(self, i):
        return self.train_feat[i, :].astype(np.float32), self.train_label[i].astype(np.int32)


    def next_feat_test(self, i):
        return self.test_feat[i, :].astype(np.float32), self.test_label[i].astype(np.int32)


    def next_test(self):

        return
