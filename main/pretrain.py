import os
import argparse

import numpy as np
import torch
import pandas as pd

from models.model_TransE import TransE
from models.model_ComplEx import ComplEx
from utils.reader import Reader
from utils.writer import Writer
from utils.logger import Logger
from utils.corrupter import Corrupter
from utils.saver import Saver


def bool_parser(s):
    if s not in {"True", "False"}:
        raise ValueError("Not a valid boolean string")
    return s == "True"


parser = argparse.ArgumentParser()
parser.add_argument("--dataset_path", type=str, default="../data/raw")
parser.add_argument("--save_path", type=str, default="./save")
parser.add_argument("--dataset", type=str, default="WN18")
parser.add_argument("--seed", type=int, default=12345)
parser.add_argument("--debug", type=bool_parser, default=False, help="debug mode")

parser.add_argument("--gpu", type=str, default="4", help="The GPU to be used")
parser.add_argument("--dim", type=int, default=64)
parser.add_argument("--epochs", type=int, default=120)
parser.add_argument("--bs", type=int, default=2048, help="batch size")
parser.add_argument("--init_lr", type=float, default=0.01)
parser.add_argument("--lr_decay", type=float, default=1.0)
parser.add_argument("--bern", type=bool_parser, default=False,
                    help="The strategy for sampling corrupt triplets. bern: bernoulli distribution.")
parser.add_argument("--margin", type=float, default=1.0)
parser.add_argument("--norm", type=int, default=2, help="[1 | 2]")
parser.add_argument("--log", type=bool_parser, default=True, help="logging or not")
parser.add_argument("--model", type=str, default="TransE", help="The model for training")
parser.add_argument("--loss", type=str, default="margin", help="loss function")
parser.add_argument("--hidden", type=int, default=100, help="hidden layer")
parser.add_argument("--neg_ratio", type=int, default=1, help="the ratio between the number of negative and positive samples")
parser.add_argument("--reg", type=float, default=0., help="The coefficient for regularization")
configs = parser.parse_args()

dataset_name = configs.dataset
bern = configs.bern
epochs = configs.epochs
batch_size = configs.bs
learning_rate = configs.init_lr
dim = configs.dim
margin = configs.margin
lr_decay = configs.lr_decay
norm = configs.norm
gpu = configs.gpu
loss_function = configs.loss
hidden = configs.hidden
reg = configs.reg

if configs.debug:
    print(
        "loaded parameters dataset_name: %s, bern: %s, epochs: %d, batch_size: %d, learning_rate: %f, dim: %d, margin: %f, lr_decay: %f, loss_function: %s, hidden: %s" %
        (dataset_name, bern, epochs, batch_size, learning_rate, dim, margin, lr_decay, loss_function, hidden))

device = torch.device("cuda")
os.environ["CUDA_VISIBLE_DEVICES"] = gpu

reader = Reader(configs)

n_train = reader.n_train
n_ent = reader.n_ent
n_rel = reader.n_rel
stat = reader.stat

corrupter = Corrupter(configs, n_ent, stat)

### create models and optimizer
if configs.model == "TransE":
    model = TransE(configs, n_ent, n_rel).to(device)
elif configs.model == "ComplEx":
    model = ComplEx(configs, n_ent, n_rel).to(device)
else:
    raise NameError("Module (%s) not found..." % configs.model)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

### training the triplets in train_data
total_loss = 0

for epoch in range(1, epochs + 1):
    if epoch % 10 == 0:
        learning_rate /= lr_decay
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    reader.shuffle()
    for i in range(0, n_train, batch_size):
        optimizer.zero_grad()
        end = i + batch_size if i + batch_size <= n_train else n_train
        pos_samples = reader.next_pos_batch(i, end)
        if configs.model == "TransE":
            neg_samples = corrupter.unif_corrupt(pos_samples)
            pos_samples, neg_samples = torch.tensor(pos_samples).to(device), torch.tensor(neg_samples).to(device)
            loss = model(pos_samples, neg_samples)
        else:
            neg_heads, neg_tails, neg_rels = corrupter.bern_corrupt_multi(pos_samples)
            neg_heads, neg_tails, neg_rels = torch.tensor(neg_heads).to(device), torch.tensor(neg_tails).to(device),\
                torch.tensor(neg_rels).to(device)
            loss = model(neg_heads, neg_tails, neg_rels)
        loss.backward()
        optimizer.step()
        total_loss += loss

    if epoch % 20 == 0:
        print("epoch %3d: lr: %.4f average loss per batch: %.4f" %
              (epoch, learning_rate, total_loss / (n_train // batch_size)), flush=True)
    total_loss = 0

### evaluate the triples in test_data
all_triplets = reader.get_all_triplets()


# triplet: .type: np.array .shape: (3,)
def rank(triplet):

    # predict tail
    new_triplet = triplet.copy().tolist()
    heads = torch.tensor(np.tile(triplet[0], n_ent)).to(device)
    tails = torch.tensor(np.arange(n_ent)).to(device)
    rels = torch.tensor(np.tile(triplet[-1], n_ent)).to(device)

    d = -model.get_score(heads, tails, rels, False)
    sorted_d_indices = d.sort(descending=False).indices
    tail_raw_ranking = np.where(sorted_d_indices.cpu().numpy() == triplet[1])[0][0].tolist() + 1
    tail_filtered_ranking = tail_raw_ranking
    for i in range(tail_raw_ranking - 1):
        new_triplet[1] = sorted_d_indices[i].item()
        if tuple(new_triplet) in all_triplets:
            tail_filtered_ranking -= 1

    # predict head
    new_triplet = triplet.copy().tolist()
    heads = torch.tensor(np.arange(n_ent)).to(device)
    tails = torch.tensor(np.tile(triplet[1], n_ent)).to(device)
    rels = torch.tensor(np.tile(triplet[-1], n_ent)).to(device)

    d = -model.get_score(heads, tails, rels, False)
    sorted_d_indices = d.sort(descending=False).indices
    head_raw_ranking = np.where(sorted_d_indices.cpu().numpy() == triplet[0])[0][0].tolist() + 1
    head_filtered_ranking = head_raw_ranking
    for i in range(head_raw_ranking - 1):
        new_triplet[0] = sorted_d_indices[i].item()
        if tuple(new_triplet) in all_triplets:
            head_filtered_ranking -= 1

    return tail_raw_ranking, tail_filtered_ranking, head_raw_ranking, head_filtered_ranking


@torch.no_grad()
def evaluate():
    ranks = []
    for triplet in reader.test_data:
        ranks.append(rank(triplet))
    ranks = np.array(ranks)
    mean_rank = ranks.mean(axis=0, dtype=np.long)
    hit10 = np.sum(ranks <= 10, axis=0) / len(ranks)
    mrr_sum = (1. / ranks).sum(axis=0)
    mrr = np.tile(np.array([mrr_sum[0] + mrr_sum[2], mrr_sum[1] + mrr_sum[3]]) / (2 * len(ranks)), 2)
    result = pd.DataFrame({"mrr": mrr, "mean rank": mean_rank, "hit10": hit10},
                          index=["tail: raw ranking", "tail: filtered ranking", "head: raw ranking", "head: filtered ranking"])
    result["hit10"] = result["hit10"].apply(lambda x: "%.2f%%" % (x * 100))
    ranks = pd.DataFrame(ranks, columns=["tail:raw", "tail:filtered", "head:raw", "head:filtered"])
    return ranks, result


model.eval()
ranks, result = evaluate()

writer = Writer(configs)
logger = Logger(configs)
saver = Saver(configs)
writer.write(result)
saver.save(model)

if configs.log:
    logger.write(ranks)
