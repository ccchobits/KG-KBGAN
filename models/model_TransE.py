import math

import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda")


class TransE(nn.Module):
    def __init__(self, configs, n_ent, n_rel):
        super(TransE, self).__init__()
        self.configs = configs
        self.depth = configs.dim
        self.margin = configs.margin
        self.norm = configs.norm
        self.ent_embed = nn.Embedding(n_ent, self.depth)
        self.rel_embed = nn.Embedding(n_rel, self.depth)
        self.initialize()

        self.all_params = [self.ent_embed, self.rel_embed]

    def initialize(self):
        self.ent_embed.weight.data.uniform_(-6 / math.sqrt(self.depth), 6 / math.sqrt(self.depth))
        self.rel_embed.weight.data.uniform_(-6 / math.sqrt(self.depth), 6 / math.sqrt(self.depth))
        # nn.init.xavier_normal_(self.ent_embed.weight.data)
        # nn.init.xavier_normal_(self.rel_embed.weight.data)
        self.rel_embed.weight.data = F.normalize(self.rel_embed.weight.data, dim=1)

    # score of TransE: for positive samples, the less the better.
    def get_score(self, heads, tails, rels, clamp=True):
        # shape: (batch_size, depth)
        heads, tails, rels = self.ent_embed(heads), self.ent_embed(tails), self.rel_embed(rels)
        # return shape: (batch_size,)
        return torch.norm(heads + rels - tails, p=self.norm, dim=-1)

    def forward(self, pos_x, neg_x):
        self.constraint()
        # shape: (batch_size,)
        pos_heads, pos_tails, pos_rels = pos_x[:, 0], pos_x[:, 1], pos_x[:, 2]
        neg_heads, neg_tails, neg_rels = neg_x[:, 0], neg_x[:, 1], neg_x[:, 2]
        pos_score = self.get_score(pos_heads, pos_tails, pos_rels)
        neg_score = self.get_score(neg_heads, neg_tails, neg_rels)
        return torch.max((self.margin + pos_score - neg_score), torch.tensor([0.]).to(device)).mean()

    def constraint(self):
        self.ent_embed.weight.data = F.normalize(self.ent_embed.weight.data, dim=1)

