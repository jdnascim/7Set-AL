import sys
sys.path.append("../")

from arch.karate_graph import *

from torch_geometric.utils import degree

from gnn_utils import *
import numpy as np
import torch
import tqdm
import copy
import json

TRAIN_SIZE = 18
VAL_SIZE = 12
VAL = 'REPR'

torch.manual_seed(12)
np.random.seed(12)

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')


with open("args.json", "r") as fp:
    args = json.load(fp)

pyg_graph_total = load_image_data(16)
pyg_graph_train = load_image_data(10, hold_test=True)


n = pyg_graph_train.num_nodes

# training set
deg = degree(pyg_graph_train.edge_index[1], n)
partition_nodes = np.argpartition(deg, -1 * TRAIN_SIZE)
train_ix = partition_nodes[-1 * TRAIN_SIZE:]

pyg_graph_train.train_mask = np.zeros(n, np.bool_)
pyg_graph_train.train_mask[train_ix] = True

# validation set
if VAL == "RANDOM":
    val_ix = np.random.choice(partition_nodes[:1 * TRAIN_SIZE], 12, replace=False)
elif VAL == "REPR":
    partition_nodes = np.argpartition(deg, -1 * (TRAIN_SIZE + VAL_SIZE))
    val_ix = partition_nodes[-1 * (TRAIN_SIZE + VAL_SIZE):-1 * TRAIN_SIZE]

pyg_graph_train.val_mask = np.zeros(n, np.bool_)
pyg_graph_train.val_mask[val_ix] = True


# test set
pyg_graph_train.test_mask = np.ones(n, np.bool_)
pyg_graph_train.test_mask[train_ix] = False
pyg_graph_train.test_mask[val_ix] = False

pyg_graph_total = pyg_graph_total.to(device)
pyg_graph_train = pyg_graph_train.to(device)

model = KarateSAGE(960, args['hidden_dim'], 2).to(device) 

run_base(model, pyg_graph_train, args, pyg_graph_total, select_best='val')