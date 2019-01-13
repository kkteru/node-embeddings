import logging
import torch
import torch.nn as nn


TOTAL_ENTITIES = 14541
TOTAL_RELATIONS = 237


class TransE(nn.Module):
    def __init__(self, params):
        super(TransE, self).__init__()
        self.params = params
        self.ent_embeddings = nn.Embedding(TOTAL_ENTITIES, self.params.embedding_dim, max_norm=1)
        self.rel_embeddings = nn.Embedding(TOTAL_RELATIONS, self.params.embedding_dim)
        self.criterion = nn.MarginRankingLoss(self.params.margin, reduction='sum')

        self.init_weights()

        logging.info('Initialized the model successfully!')

    def init_weights(self):
        nn.init.xavier_uniform_(self.ent_embeddings.weight.data)
        nn.init.xavier_uniform_(self.rel_embeddings.weight.data)

    def get_score(self, h, t, r):
        return torch.norm(h + r - t, self.params.p_norm, -1)

    def get_positive_score(self, score):
        return score[0: int(len(score) / 2)]

    def get_negative_score(self, score):
        return score[int(len(score) / 2): len(score)]

    def compute_loss(self, pos_score, neg_score):
        y = torch.Tensor([-1])
        return self.criterion(pos_score, neg_score, y)

    def forward(self, batch_h, batch_t, batch_r):
        h = self.ent_embeddings(torch.from_numpy(batch_h))
        t = self.ent_embeddings(torch.from_numpy(batch_t))
        r = self.rel_embeddings(torch.from_numpy(batch_r))

        score = self.get_score(h, t, r)

        pos_score = self.get_positive_score(score)
        neg_score = self.get_negative_score(score)

        return self.compute_loss(pos_score, neg_score)
