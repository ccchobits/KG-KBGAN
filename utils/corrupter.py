import numpy as np


class Corrupter:
    def __init__(self, configs, n_ent, stat):
        super(Corrupter, self).__init__()
        self.configs = configs
        self.n_ent = n_ent
        self.stat = stat

    def unif_corrupt(self, pos_samples):
        size = len(pos_samples)
        new_ent = np.random.randint(low=0, high=self.n_ent, size=(size,))
        head_or_tail = np.random.randint(low=0, high=2, size=(size,))
        neg_samples = np.copy(pos_samples)
        for i in range(size):
            if head_or_tail[i] == 0:
                neg_samples[i][0] = new_ent[i]
            else:
                neg_samples[i][1] = new_ent[i]
        return neg_samples

    def bern_corrupt(self, pos_samples):
        size = len(pos_samples)
        new_ent = np.random.randint(low=0, high=self.n_ent, size=(size,))
        head_or_tail = np.empty(size)
        rand = np.random.random(size)
        for i in range(size):
            if rand[i] < self.stat[pos_samples[i][2]][0]:
                head_or_tail[i] = 1
            else:
                head_or_tail[i] = 0
        neg_samples = np.copy(pos_samples)
        for i in range(size):
            if head_or_tail[i] == 0:
                neg_samples[i][0] = new_ent[i]
            else:
                neg_samples[i][1] = new_ent[i]
        return neg_samples

    def bern_corrupt_multi(self, pos_samples, keep_truth=True):
        size = len(pos_samples)
        n_samples = 20
        pos_heads, pos_tails, pos_rels = pos_samples[:, 0], pos_samples[:, 1], pos_samples[:, 2]
        neg_heads = np.tile(pos_heads[:, np.newaxis], (1, n_samples))
        neg_tails = np.tile(pos_tails[:, np.newaxis], (1, n_samples))
        neg_rels = np.tile(pos_rels[:, np.newaxis], (1, n_samples))

        rand = np.random.random(size)
        if keep_truth:
            new_ents = np.random.choice(self.n_ent, (size, n_samples - 1))
            for i in range(size):
                if rand[i] < self.stat[pos_samples[i][2]][0]:
                    neg_tails[i, 1:] = new_ents[i]
                else:
                    neg_heads[i, 1:] = new_ents[i]
        else:
            new_ents = np.random.choice(self.n_ent, (size, n_samples))
            for i in range(size):
                if rand[i] < self.stat[pos_samples[i][2]][0]:
                    neg_tails[i, :] = new_ents[i]
                else:
                    neg_heads[i, :] = new_ents[i]
        return neg_heads, neg_tails, neg_rels