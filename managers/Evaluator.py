import numpy as np
import torch
from sklearn.metrics.pairwise import pairwise_distances
import pdb


class Evaluator():
    def __init__(self, model, data_sampler, sample_size=0):
        self.model = model
        self.data_sampler = data_sampler
        self.sample_size = sample_size

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
        hit10 = len(isHit10ListHead) / len(rankListHead)

        assert len(rankListHead) == len(self.data_sampler.data)

        log_data = dict([
            ('hit@10', hit10),
            # ('mrr', mrr),
            ('mr', mr)])

        return log_data
