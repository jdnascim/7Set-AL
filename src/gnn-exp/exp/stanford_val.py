import sys
sys.path.append("../")

from arch.stanford_graph import StanfordGraph

from gnn_utils import *
import numpy as np
import torch
import tqdm
import copy

torch.manual_seed(13)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

args = {
    'device': device,
    'hidden_dim': 2048,
    'lr': 1e-4,
    'epochs': 100,
    'weight_decay':1e-3
}

pyg_graph = load_image_data(16)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

pyg_graph = pyg_graph.to(device)

model = StanfordGraph(960, args['hidden_dim'], 2, 3, 0.5, device).to(device) 

run_base(mode, pyg_graph, args)