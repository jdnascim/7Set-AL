import sys
sys.path.append("../")

import arch.karate_graph

from gnn_utils import *
import numpy as np
import torch
import tqdm
import copy
import json

seeds = np.array([12, 13, 16, 18, 21, 23, 29, 40, 50, 65])
results_bacc = np.zeros_like(seeds, dtype=np.float32)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open("args.json", "r") as fp:
    args = json.load(fp)

pyg_graph_total = load_image_data(16)
pyg_graph_train = load_image_data(10, hold_test=True)

for i, s in enumerate(seeds):
    print("{}/{}".format(i,len(seeds)))

    args['random_state'] = s

    torch.manual_seed(s)
    np.random.seed(s)

    pyg_graph_train = shuffle_data(pyg_graph_train, args)
    
    n = pyg_graph_train.num_nodes
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    pyg_graph_total = pyg_graph_total.to(device)
    pyg_graph_train = pyg_graph_train.to(device)
    
    model = getattr(arch.karate_graph, args["model"])(960, args['hidden_dim'], 2).to(device) 
    
    _, results_bacc[i] = run_base(model, pyg_graph_train, args, pyg_graph_total)

print("Mean: {}".format(results_bacc.mean()))
print("Std: {}".format(results_bacc.std()))

