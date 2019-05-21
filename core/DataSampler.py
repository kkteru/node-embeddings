import os
import pickle as pkl

import logging
import random
import numpy as np
import pdb
from scipy.sparse import csc_matrix


def get_all_adj(adj_list):
    '''
    adj_list: List of sparse adjacency matrices
    '''
    rows = []
    cols = []
    dats = []
    dim = adj_list[0].shape
    for adj in adj_list:
        rows += adj.tocoo().row.tolist()
        cols += adj.tocoo().col.tolist()
        dats += adj.tocoo().data.tolist()
    row = np.array(rows)
    col = np.array(cols)
    data = np.array(dats)
    return csc_matrix((data, (row, col)), shape=dim)


def sample_neg(adj_list, train_triplets, valid_triplets, test_triplets, max_train_num=None):

    train_pos = (train_triplets[:, 0], train_triplets[:, 1], train_triplets[:, 2])
    valid_pos = (valid_triplets[:, 0], valid_triplets[:, 1], valid_triplets[:, 2])
    test_pos = (test_triplets[:, 0], test_triplets[:, 1], test_triplets[:, 2])
    # if max_train_num is set, randomly sample train links
    if max_train_num is not None:
        perm = np.random.permutation(len(train_pos[0]))[:max_train_num]
        train_pos = (train_pos[0][perm], train_pos[1][perm], train_pos[2][perm])
    # sample negative links for train/test
    train_num, valid_num, test_num = len(train_pos[0]), len(valid_pos[0]), len(test_pos[0])
    neg = ([], [], [])
    adj_acc = get_all_adj(adj_list)  # Use this to add positive examples of no-link, if need be
    n = adj_acc.shape[0]
    r = len(adj_list)
    print('sampling negative links for train and test')
    while len(neg[0]) < train_num + valid_num + test_num:
        i, j, k = random.randint(0, n - 1), random.randint(0, n - 1), random.randint(0, r - 1)
        if i != j and adj_list[k][i, j] == 0:
            neg[0].append(i)
            neg[1].append(j)
            neg[2].append(k)
        # else add negative examples of no-link, if need be
        else:
            continue
    train_neg = (np.array(neg[0][:train_num]), np.array(neg[1][:train_num]), np.array(neg[2][:train_num]))
    valid_neg = (np.array(neg[0][train_num:train_num + valid_num]), np.array(neg[1][train_num:train_num + valid_num]), np.array(neg[2][train_num:train_num + valid_num]))
    test_neg = (np.array(neg[0][train_num + valid_num:]), np.array(neg[1][train_num + valid_num:]), np.array(neg[2][train_num + valid_num:]))
    return train_pos, train_neg, valid_pos, valid_neg, test_pos, test_neg


class DataSampler():
    def __init__(self, params):

        data_path = os.path.join(params.main_dir, 'data/{}/{}.pickle'.format(params.dataset, params.dataset))

        with open(data_path, 'rb') as f:
            data = pkl.load(f)
        adj_list = data['adj_list']
        train_triplets = data['train_triplets']
        valid_triplets = data['valid_triplets']
        test_triplets = data['test_triplets']

        params.total_ent = adj_list[0].shape[0]
        params.total_rel = len(adj_list)

        self.train_pos, self.train_neg, self.valid_pos, self.valid_neg, self.test_pos, self.test_neg = sample_neg(adj_list, train_triplets, valid_triplets, test_triplets)

        self.train_idx = np.arange(len(self.train_pos[0]))
        self.batch_size = len(self.train_pos[0]) // params.nBatches

    def get_batch(self, n_batch):

        if n_batch == 0:
            np.random.shuffle(self.train_idx)

        ids = self.train_idx[n_batch * self.batch_size: (n_batch + 1) * self.batch_size]

        batch_h = np.concatenate((self.train_pos[0][ids], self.train_neg[0][ids]))
        batch_t = np.concatenate((self.train_pos[1][ids], self.train_neg[1][ids]))
        batch_r = np.concatenate((self.train_pos[2][ids], self.train_neg[2][ids]))
        batch_y = np.concatenate((np.ones(self.batch_size), -1 * np.ones(self.batch_size)))

        return batch_h, batch_t, batch_r, batch_y

    def get_valid_data(self):

        batch_h = np.concatenate((self.valid_pos[0], self.valid_neg[0]))
        batch_t = np.concatenate((self.valid_pos[1], self.valid_neg[1]))
        batch_r = np.concatenate((self.valid_pos[2], self.valid_neg[2]))
        batch_y = np.concatenate((np.ones(len(self.valid_pos[0])), -1 * np.ones(len(self.valid_neg[0]))))

        return batch_h, batch_t, batch_r, batch_y

    def get_test_data(self):

        batch_h = np.concatenate((self.test_pos[0], self.test_neg[0]))
        batch_t = np.concatenate((self.test_pos[1], self.test_neg[1]))
        batch_r = np.concatenate((self.test_pos[2], self.test_neg[2]))
        batch_y = np.concatenate((np.ones(len(self.test_pos[0])), -1 * np.ones(len(self.test_neg[0]))))

        return batch_h, batch_t, batch_r, batch_y
