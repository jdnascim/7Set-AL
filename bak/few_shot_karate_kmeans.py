import sys
sys.path.append("../")

from arch.karate_graph import KarateGraph

from gnn_utils import *
import numpy as np
import torch
import tqdm
import copy


torch.manual_seed(13)
np.random.seed(13)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

args = {
    'device': device,
    'hidden_dim': 4096,
    'lr': 1e-5,
    'epochs': 40,
    'weight_decay':1e-3,
    'cross_validation': True,
    'train_size': 18,
    'val_size':12
}

pyg_graph_total = load_image_data(16)
pyg_graph_train = load_image_data(16, hold_test=True)

n = pyg_graph_train.num_nodes
y = pyg_graph_train.y

ix = select(pyg_graph_train.x, 30)

train_ix, val_ix, _, _ = train_test_split(ix, y[ix], train_size=18, stratify=y[ix])

pyg_graph_train.train_mask = np.zeros(n, np.bool_)
pyg_graph_train.train_mask[train_ix] = True
pyg_graph_train.val_mask = np.zeros(n, np.bool_)
pyg_graph_train.val_mask[val_ix] = True
pyg_graph_train.test_mask = np.ones(n, np.bool_)
pyg_graph_train.test_mask[ix] = False

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

pyg_graph_total = pyg_graph_total.to(device)
pyg_graph_train = pyg_graph_train.to(device)

model = KarateGraph(960, args['hidden_dim'], 2).to(device) 

run_base(model, pyg_graph_train, args, pyg_graph_total)