import logging
import torch
import torch.nn as nn


TOTAL_ENTITIES = 14951
TOTAL_RELATIONS = 1345


class TransE(nn.Module):
    def __init__(self, params):
        super(TransE, self).__init__()
        self.params = params
        self.ent_embeddings = nn.Embedding(TOTAL_ENTITIES, self.params.embedding_dim, max_norm=1)
        self.rel_embeddings = nn.Embedding(TOTAL_RELATIONS, self.params.embedding_dim)

        self.init_weights()

        logging.info('Initialized the model successfully!')

    def init_weights(self):
        nn.init.xavier_uniform_(self.ent_embeddings.weight.data)
        nn.init.xavier_uniform_(self.rel_embeddings.weight.data)

    def get_score(self, h, t, r):
        return torch.norm(h + r - t, self.params.p_norm, -1)

    def forward(self, batch_h, batch_t, batch_r):
        h = self.ent_embeddings(torch.from_numpy(batch_h).to(device=self.params.device))
        t = self.ent_embeddings(torch.from_numpy(batch_t).to(device=self.params.device))
        r = self.rel_embeddings(torch.from_numpy(batch_r).to(device=self.params.device))

        return self.get_score(h, t, r)
