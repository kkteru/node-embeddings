import logging
import torch
import torch.nn as nn


class ComplEx(nn.Module):
    def __init__(self, params):
        super(ComplEx, self).__init__()
        self.params = params

        self.ent_re_embeddings = nn.Embedding(
            self.params.total_ent, self.params.embedding_dim
        )
        self.ent_im_embeddings = nn.Embedding(
            self.params.total_ent, self.params.embedding_dim
        )
        self.rel_re_embeddings = nn.Embedding(
            self.params.total_ent, self.params.embedding_dim
        )
        self.rel_im_embeddings = nn.Embedding(
            self.params.total_ent, self.params.embedding_dim
        )
        # self.criterion = nn.Softplus()
        self.criterion = nn.MarginRankingLoss(self.params.margin, reduction='sum')
        self.init_weights()

        logging.info('Initialized the model successfully!')

    def init_weights(self):
        nn.init.xavier_uniform(self.ent_re_embeddings.weight.data)
        nn.init.xavier_uniform(self.ent_im_embeddings.weight.data)
        nn.init.xavier_uniform(self.rel_re_embeddings.weight.data)
        nn.init.xavier_uniform(self.rel_im_embeddings.weight.data)

    def get_score(self, h_re, h_im, t_re, t_im, r_re, r_im):
        return -torch.sum(
            h_re * t_re * r_re
            + h_im * t_im * r_re
            + h_re * t_im * r_im
            - h_im * t_re * r_im,
            -1,
        )

    def forward(self, batch_h, batch_t, batch_r, batch_y):
        h_re = self.ent_re_embeddings(torch.from_numpy(batch_h))
        h_im = self.ent_im_embeddings(torch.from_numpy(batch_h))
        t_re = self.ent_re_embeddings(torch.from_numpy(batch_t))
        t_im = self.ent_im_embeddings(torch.from_numpy(batch_t))
        r_re = self.rel_re_embeddings(torch.from_numpy(batch_r))
        r_im = self.rel_im_embeddings(torch.from_numpy(batch_r))

        y = torch.from_numpy(batch_y).type(torch.FloatTensor)

        score = self.get_score(h_re, h_im, t_re, t_im, r_re, r_im)

        pos_score = score[0: int(len(score) / 2)]
        neg_score = score[int(len(score) / 2): len(score)]

        regul = (
            torch.mean(h_re ** 2)
            + torch.mean(h_im ** 2)
            + torch.mean(t_re ** 2)
            + torch.mean(t_im ** 2)
            + torch.mean(r_re ** 2)
            + torch.mean(r_im ** 2)
        )
        # loss = torch.mean(self.criterion(score * y)) + self.params.lmbda * regul
        loss = self.criterion(pos_score, neg_score, torch.Tensor([-1]))
        return loss, pos_score, neg_score
