"""
   Contains all the utility functions that would be needed
   1. _normalized
   2. _split
   3._batchify
   4. get_batches
   """


import torch
import numpy as np;
from torch.autograd import Variable


def normal_std(x):
    return x.std() * np.sqrt((len(x) - 1.) / (len(x)))


class Data_utility(object):
    # train and valid is the ratio of training set and validation set. test = 1 - train - valid
    def __init__(self, file_name, train, valid, cuda, horizon, window, normalize=2):
        self.cuda = cuda;
        self.window_length = window;
        self.horizon = horizon
        fin = open(file_name);
        self.original_data = np.loadtxt(fin, delimiter=',');
        self.normalized_data = np.zeros(self.original_data.shape);
        self.original_rows, self.original_columns = self.normalized_data.shape;
        self.normalize = 2
        self.scale = np.ones(self.original_columns);
        self._normalized(normalize);

        #after this step train, valid and test have the respective data, split from original_data according to the ratios
        self._split(int(train * self.original_rows), int((train + valid) * self.original_rows), self.original_rows);

        self.scale = torch.from_numpy(self.scale).float();
        tmp = self.test[1] * self.scale.expand(self.test[1].size(0), self.original_columns);

        if self.cuda:
            self.scale = self.scale.cuda();
        self.scale = Variable(self.scale);

        #rse and rae must be some sort of errors for now, will come back to them later
        self.rse = normal_std(tmp);
        self.rae = torch.mean(torch.abs(tmp - torch.mean(tmp)));

    def _normalized(self, normalize):
        # normalized by the maximum value of entire matrix.

        if (normalize == 0):
            self.normalized_data = self.original_data

        if (normalize == 1):
            self.normalized_data = self.original_data / np.max(self.original_data);

        # normalized by the maximum value of each row(sensor).
        if (normalize == 2):
            for i in range(self.original_columns):
                self.scale[i] = np.max(np.abs(self.original_data[:, i]));
                self.normalized_data[:, i] = self.original_data[:, i] / np.max(np.abs(self.original_data[:, i]));

    def _split(self, train, valid, test):

        train_set = range(self.window_length + self.horizon - 1, train);
        valid_set = range(train, valid);
        test_set = range(valid, self.original_rows);
        self.train = self._batchify(train_set, self.horizon);
        self.valid = self._batchify(valid_set, self.horizon);
        self.test = self._batchify(test_set, self.horizon);

    def _batchify(self, idx_set, horizon):

        n = len(idx_set);
        X = torch.zeros((n, self.window_length, self.original_columns));
        Y = torch.zeros((n, self.original_columns));

        for i in range(n):
            end = idx_set[i] - self.horizon + 1;
            start = end - self.window_length;
            X[i, :, :] = torch.from_numpy(self.normalized_data[start:end, :]);
            Y[i, :] = torch.from_numpy(self.normalized_data[idx_set[i], :]);

        """
            Here matrix X is 3d matrix where each of it's 2d matrix is the separate window which has to be sent in for training.
            Y is validation.           
        """
        return [X, Y];

    def get_batches(self, inputs, targets, batch_size, shuffle=True):
        length = len(inputs)
        if shuffle:
            index = torch.randperm(length)
        else:
            index = torch.LongTensor(range(length))
        start_idx = 0
        while (start_idx < length):
            end_idx = min(length, start_idx + batch_size)
            excerpt = index[start_idx:end_idx]
            X = inputs[excerpt];
            Y = targets[excerpt];
            if (self.cuda):
                X = X.cuda();
                Y = Y.cuda();
            yield Variable(X), Variable(Y);
            start_idx += batch_size