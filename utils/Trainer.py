class Trainer():
    def __init__(self, model, optimizer, data, params):
        self.model = model
        self.data = data
        self.optimizer = optimizer  # Correct this!

        self.best_hit_10 = 0
        self.bad_count = 0

    def one_step():  # rename
        batch_h, batch_t, batch_r = self.data.get_batch(params.batch_size)
        loss = self.model(batch_h, batch_t, batch_r)
        self.optimizer.zero_grad()
        loss.autograd()
        self.optmizer.step()

    def select_model(to_log):
        pass
