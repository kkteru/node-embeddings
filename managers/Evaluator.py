import numpy as np
import torch
import pdb
from sklearn import metrics
import torch.nn.functional as F


def get_torch_sparse_matrix(A, dev):
    '''
    A : Sparse adjacency matrix
    '''
    idx = torch.LongTensor([A.tocoo().row, A.tocoo().col])
    dat = torch.FloatTensor(A.tocoo().data)
    return torch.sparse.FloatTensor(idx, dat, torch.Size([A.shape[0], A.shape[1]])).to(device=dev)


class Evaluator():
    def __init__(self, model, data_sampler, params):
        self.model = model
        self.data_sampler = data_sampler
        self.params = params

    def get_log_data(self, data='valid'):

        if data == 'valid':
            eval_batch_h, eval_batch_t, eval_batch_r = self.data_sampler.get_valid_data()
        elif data == 'test':
            eval_batch_h, eval_batch_t, eval_batch_r = self.data_sampler.get_test_data()

        score = self.model(eval_batch_h, eval_batch_t, eval_batch_r)

        all_pos_scores = score[0: int(len(score) / 2)].detach().cpu().tolist()
        all_neg_scores = score[int(len(score) / 2): len(score)].detach().cpu().tolist()

        all_labels = [0] * len(all_pos_scores) + [1] * len(all_neg_scores)
        auc = metrics.roc_auc_score(all_labels, all_pos_scores + all_neg_scores)

        log_data = dict([
            ('auc', auc)])

        return log_data
