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

        distHead = pairwise_distances(c_h_e, self.model.ent_embeddings.weight.data.cpu().numpy(), metric='manhattan')

        rankArrayHead = np.argsort(distHead, axis=1)

        # Don't check whether it is false negative
        rankListHead = [int(np.argwhere(elem[1] == elem[0])) for elem in zip(self.data_sampler.data[:, 0], rankArrayHead)]

        isHit10ListHead = [x for x in rankListHead if x < 10]

        assert len(rankListHead) == len(self.data_sampler.data)

        mr_h = np.mean(rankListHead)
        hit10_h = len(isHit10ListHead) / len(rankListHead)

# -------------------------------------------------------------------- #

        c_t_e = h_e + r_e

        distTail = pairwise_distances(c_t_e, self.model.ent_embeddings.weight.data.cpu().numpy(), metric='manhattan')

        rankArrayTail = np.argsort(distTail, axis=1)

        # Don't check whether it is false negative
        rankListTail = [int(np.argwhere(elem[1] == elem[0])) for elem in zip(self.data_sampler.data[:, 1], rankArrayTail)]

        isHit10ListTail = [x for x in rankListTail if x < 10]

        assert len(rankListTail) == len(self.data_sampler.data)

        mr_t = np.mean(rankListTail)
        hit10_t = len(isHit10ListTail) / len(rankListTail)

        mr = (mr_h + mr_t) / 2
        hit10 = (hit10_h + hit10_t) / 2

        log_data = dict([
            ('hit@10', hit10),
            ('mr', mr)])

        return log_data
