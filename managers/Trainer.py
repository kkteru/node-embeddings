import pdb
import os
import logging
import torch
import torch.optim as optim
import torch.nn as nn


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

        self.best_mr = 1e10
        self.last_mr = 1e10
        self.bad_count = 0

        assert self.optimizer is not None

    def one_step(self, n_batch):  # rename
        batch_h, batch_t, batch_r = self.data.get_batch(n_batch)
        score = self.model(batch_h, batch_t, batch_r)

        pos_score = score[0: int(len(score) / 2)]
        neg_score = score[int(len(score) / 2): len(score)]

        loss = self.criterion(pos_score, neg_score, torch.Tensor([-1]).to(device=self.params.device))
#        pdb.set_trace()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss

    def select_model(self, log_data):
        if log_data['mr'] < self.best_mr:
            self.bad_count = 0
            torch.save(self.model, os.path.join(self.params.exp_dir, 'best_model.pth'))  # Does it overwrite or fuck with the existing file?
            logging.info('Better model found w.r.t MR. Saved it!')
            self.best_mr = log_data['mr']
        else:
            self.bad_count = self.bad_count + 1
            if self.bad_count > self.params.patience:
                logging.info('Out of patience. Stopping the training loop.')
                return False
        self.last_mr = log_data['mr']
        return True
