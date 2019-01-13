import os
import logging
import torch
import torch.optim as optim


class Trainer():
    def __init__(self, model, data, params):
        self.model = model
        self.data = data
        self.optimizer = None
        self.params = params

        if params.optimizer == "SGD":
            self.optimizer = optim.SGD(self.model.parameters(), lr=params.lr, momentum=params.momentum)

        self.best_mrr = 0
        self.last_mrr = 0
        self.bad_count = 0

        assert self.optimizer != None

    def one_step(self, batch_size):  # rename
        batch_h, batch_t, batch_r = self.data.get_batch(batch_size)
        loss = self.model(batch_h, batch_t, batch_r)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss

    def select_model(self, log_data):
        if log_data['mrr'] > self.last_mrr:
            self.last_mrr = log_data['mrr']
            self.bad_count = 0

            if log_data['mrr'] > self.best_mrr:
                torch.save(self.model, os.path.join(self.params.exp_dir, 'best_model.pth'))  # Does it overwrite or fuck with the existing file?
                logging.info('Better model found w.r.t MRR. Saved it!')
                self.best_mrr = log_data['mrr']
        else:
            self.bad_count = self.bad_count + 1
            if self.bad_count > self.params.patience:
                logging.info('Out of patience. Stopping the training loop.')
                return False
        return True
