import numpy as np


class DataSampler():
    def __init__(self, file_path, debug=False):
        end = 20000 if debug else -1
        with open(file_path) as f:
            self.data = np.array([map(int, sample.split()) for sample in f.split('\n')[1:end]])
        self.totalEnt = 0 # Fill this
        self.totalRel = 0 # Fill this

    def get_batch(self, batch_size):
        ids = np.random.random_integers(0, len(self.data), batch_size)
        batch = self.data[ids]

        return batch[:, 0], batch[:, 1], batch[:, 2]

    def _sample_negative(self, h, t, r):
        _h = self.data[np.random.ranint(0, len(self.data)), 0]
        return _h, t, r

    def get_negative_batch(self, batch_h, batch_t, batch_r):
        _batch_h = np.zeros(len(batch_h))
        for i, (sample_h, sample_t, sample_r) in enumerate(zip(batch_h, batch_t, batch_r)):
            _batch_h[i], _, _ = self._sample_negative(sample_h, sample_t, sample_r)

        return _batch_h, batch_t, batch_r
