import numpy as np
import torch
from sklearn.metrics.pairwise import pairwise_distances
import pdb


class Evaluator():
    def __init__(self, model, data_sampler, sample_size=0):
        self.model = model
        self.data_sampler = data_sampler
        self.sample_size = sample_size

    def _rank(self, sample):
        if self.sample_size == 0:
            head_ids = np.array(list(self.data_sampler.ent))
        else:
            idx = np.random.random_integers(0, len(self.data_sampler.ent) - 1, self.sample_size)
            head_ids = np.array(list(self.data_sampler.ent))[idx]
            head_ids[0] = sample[0]

        eval_batch = np.array(list(zip(head_ids, [sample[1]] * len(head_ids), [sample[2]] * len(head_ids))))
        if self.model.params.filter:
            eval_batch = np.array(list(filter(lambda x: tuple(x) not in self.data_sampler.data_set, eval_batch)))  # This only filters from validation set. Wrong!
        eval_batch = torch.from_numpy(eval_batch)

        heads = self.model.ent_embeddings(eval_batch[:, 0].to(device=self.model.params.device))
        tails = self.model.ent_embeddings(eval_batch[:, 1].to(device=self.model.params.device))
        rels = self.model.rel_embeddings(eval_batch[:, 2].to(device=self.model.params.device))

        scores = self.model.get_score(heads, tails, rels)

        assert scores.shape == (len(head_ids), )

        return np.where(head_ids[scores.argsort().cpu()] == sample[0])[0][0] + 1

    def get_log_data(self):
        # pdb.set_trace()

        h_e = self.model.ent_embeddings.weight.data.cpu().numpy()[self.data_sampler.data[:, 0]]
        t_e = self.model.ent_embeddings.weight.data.cpu().numpy()[self.data_sampler.data[:, 1]]
        r_e = self.model.rel_embeddings.weight.data.cpu().numpy()[self.data_sampler.data[:, 2]]

        c_h_e = t_e - r_e

        dist = pairwise_distances(c_h_e, self.model.ent_embeddings.weight.data.cpu().numpy(), metric='manhattan')

        rankArrayHead = np.argsort(dist, axis=1)
        # Don't check whether it is false negative
        rankListHead = [int(np.argwhere(elem[1] == elem[0])) for elem in zip(self.data_sampler.data[:, 0], rankArrayHead)]

        isHit10ListHead = [x for x in rankListHead if x < 10]

        mr = np.mean(rankListHead)
        hit10 = len(isHit10ListHead)

        # ranks = np.array(list(map(self._rank, self.data_sampler.data)))

        assert len(rankListHead) == len(self.data_sampler.data)

        # hit10 = np.sum(ranks <= 10) / len(ranks)
        # mrr = np.mean(1 / ranks)
        # mr = np.mean(ranks)

        log_data = dict([
            ('hit@10', hit10),
            # ('mrr', mrr),
            ('mr', mr)])

        return log_data
