import torch
import torch.optim as optim


class Trainer():
    def __init__(self, model, data, params):
        self.model = model
        self.data = data
        self.optimizer = None

        if params.optimizer == "SGD":
            self.optimizer = optim.SGD(self.model.parameters(), lr=params.lr)

        self.best_mrr = 0
        self.bad_count = 0

        assert self.optimizer != None

    def one_step(self, batch_size):  # rename
        batch_h, batch_t, batch_r = self.data.get_batch(batch_size)
        loss = self.model(batch_h, batch_t, batch_r)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def select_model(self, log_data):
        if log_data['mrr'] > self.best_mrr:
            self.bad_count = 0
            torch.save(self.model, 'best_model.pth')
        else:
            self.bad_count = self.bad_count + 1
            if self.bad_count > self.params.patience:
                return False
        return True
