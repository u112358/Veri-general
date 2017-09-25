"""
Code ideas from https://github.com/Newmu/dcgan and tensorflow mnist dataset reader
"""
import numpy as np
import scipy.io as sio
import os.path as op
from six.moves import xrange


class FakeDataset:

    def __init__(self, data_dir, batch_size):
        """
        data_dir is the directory of the input data, which should be the format of 'XX/XX/XX/'
        total_batches is the number of the batches, which begins from 1.mat to total_batches.mat
        """
        print("Initializing Batch Dataset Reader...")
        self.batch_size = batch_size

        self.data = sio.loadmat(data_dir+'fakefeat.mat')
        self.feat = self.data['feat']
        self.index = np.squeeze(self.data['ind'], axis=0)
        self.attr = self.data['attr']

    #     process the sub-dataset for training
        total_len = np.shape(self.index)[0]
        self.total_batches = total_len / self.batch_size
        self.batch_id = 0
        self.epoch = 1

        ulabel = np.unique(self.index)
        len = np.shape(ulabel)[0]
        for i in xrange(len):
            pos = np.where(self.index==ulabel[i])
            self.index[pos] = i


    def next_batch(self):
        if self.batch_id >= self.total_batches:
            len = np.shape(self.index)[0]
            ind = np.arange(len)
            np.random.shuffle(ind)
            self.feat = self.feat[ind, :]
            self.index = self.index[ind]
            self.attr = self.attr[ind, :]
            self.batch_id = 0

            string = op.join('------------ finished', str(self.epoch))
            string = op.join(string,'epoch---------------')
            print(string)
            self.epoch += 1
        start = self.batch_id * self.batch_size
        end = (self.batch_id+1) * self.batch_size
        self.batch_id += 1
        return self.feat[start:end, :].astype(np.float32), self.index[start:end].astype(np.int32), self.attr[start:end, :].astype(np.float32)

