import torch.nn as nn

from .Model import Model


class TransE(Model):
    def __init__(self, params, totalEnt, totalRel):
        super(TransE, self).__init__()
        self.ent_embeddings = nn.Embedding(totalEnt, params.embedding_dim)
        self.rel_embeddings = nn.Embedding(totalRel, params.embedding_dim)
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform(self.ent_embeddings.weight.data)
        nn.init.xavier_uniform(self.rel_embeddings.weight.data)
