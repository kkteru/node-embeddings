import numpy as np
import torch


class Evaluator():
    def __init__(self, model, data_sampler, sample_size):
        self.model = model
        self.data_sampler = data_sampler
        self.sample_size = sample_size

    def _rank(self, sample):
        idx = np.random.random_integers(0, len(self.data_sampler.ent) - 1, self.sample_size)

        head_ids = np.array(list(self.data_sampler.ent))[idx]
        head_ids[0] = sample[0]

        heads = self.model.ent_embeddings(torch.LongTensor(head_ids))
        tails = self.model.ent_embeddings(torch.LongTensor([sample[1]] * len(head_ids)))
        rels = self.model.ent_embeddings(torch.LongTensor([sample[2]] * len(head_ids)))

        scores = self.model.get_score(heads, tails, rels)

        assert scores.shape == (len(head_ids), )

        return np.where(head_ids[scores.argsort()] == sample[0])[0][0] + 1

    def get_log_data(self):
        ranks = np.array(list(map(self._rank, self.data_sampler.data)))

        assert len(ranks) == len(self.data_sampler.data)

        hit10 = np.sum(ranks < 10) / len(ranks)
        mrr = np.mean(1 / ranks)
        mr = np.mean(ranks)

        log_data = dict([
            ('hit@10', hit10),
            ('mrr', mrr),
            ('mr', mr)])

        return log_data
