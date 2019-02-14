import logging
import random
import numpy as np
import pdb


class DataSampler():
    def __init__(self, file_path, nBatches=1, debug=False):

        end = 20001 if debug else -1
        with open(file_path) as f:
            self.data = np.array([list(map(int, sample.split())) for sample in f.read().split('\n')[1:end]], dtype=np.int64)

        assert self.data.shape[1] == 3

        self.batch_size = int(len(self.data) / nBatches)
        # pdb.set_trace()
        self.idx = np.arange(len(self.data))

        self.data_set = set(map(tuple, self.data))
        self.ent = self.get_ent(self.data)  # Fill this
        self.rel = self.get_rel(self.data)  # Fill this

        logging.info('Loaded data sucessfully from %s. Samples = %d; Total entities = %d; Total relations = %d' % (file_path, len(self.data), len(self.ent), len(self.rel)))

    def get_ent(self, debug=False):
        return set([i.item() for i in self.data[:, 0:2].reshape(-1)])

    def get_rel(self, debug=False):
        return set([i.item() for i in self.data[:, 2]])

    def get_batch(self, n_batch):
        if n_batch == 0:
            np.random.shuffle(self.idx)
        # pdb.set_trace()
        ids = self.idx[n_batch * self.batch_size: (n_batch + 1) * self.batch_size]
        pos_batch = self.data[ids]

        neg_batch = self.get_negative_batch(pos_batch)

        batch = np.concatenate((pos_batch, neg_batch), axis=0)

        return batch[:, 0], batch[:, 1], batch[:, 2]

    def _sample_negative(self, sample):
        neg_sample = np.array(sample)
        if random.random() < 0.5:
            while tuple(neg_sample) in self.data_set:
                neg_sample[0] = np.random.randint(0, len(self.ent))
        else:
            while tuple(neg_sample) in self.data_set:
                neg_sample[1] = np.random.randint(0, len(self.ent))
        return neg_sample

    def get_negative_batch(self, batch):
        neg_batch = np.zeros(batch.shape, dtype=np.int64)
        for i, sample in enumerate(batch):
            neg_batch[i] = self._sample_negative(sample)

        return neg_batch
