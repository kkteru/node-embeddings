import pdb
import os
import logging
import torch
import torch.optim as optim
import torch.nn as nn
from sklearn import metrics


class Trainer():
    def __init__(self, model, data, params):
        self.model = model
        self.data = data
        self.optimizer = None
        self.params = params

        if params.optimizer == "SGD":
            self.optimizer = optim.SGD(self.model.parameters(), lr=params.lr, momentum=params.momentum)
        if params.optimizer == "Adam":
            self.optimizer = optim.Adam(self.model.parameters(), lr=params.lr)

        self.criterion = nn.MarginRankingLoss(self.params.margin, reduction='sum')

        self.best_metric = 1e10
        self.last_metric = 1e10
        self.bad_count = 0

        assert self.optimizer is not None

    def one_epoch(self):
        all_pos_scores = []
        all_neg_scores = []
        total_loss = 0
        for b in range(self.params.nBatches):
            batch_h, batch_t, batch_r, batch_y = self.data.get_batch(b)
            loss, pos_score, neg_score = self.model(batch_h, batch_t, batch_r, batch_y)

            all_pos_scores += pos_score.detach().cpu().tolist()
            all_neg_scores += neg_score.detach().cpu().tolist()

            total_loss += loss.detach().cpu()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        all_labels = [0] * len(all_pos_scores) + [1] * len(all_neg_scores)
        auc = metrics.roc_auc_score(all_labels, all_pos_scores + all_neg_scores)

        return total_loss, auc

    def select_model(self, log_data):
        if log_data['auc'] < self.best_metric:
            self.bad_count = 0
            torch.save(self.model, os.path.join(self.params.exp_dir, 'best_model.pth'))  # Does it overwrite or fuck with the existing file?
            logging.info('Better model found w.r.t MR. Saved it!')
            self.best_mr = log_data['auc']
        else:
            self.bad_count = self.bad_count + 1
            if self.bad_count > self.params.patience:
                logging.info('Out of patience. Stopping the training loop.')
                return False
        self.last_metric = log_data['auc']
        return True
