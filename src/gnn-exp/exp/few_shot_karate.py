import sys
sys.path.append("../")

import arch.karate_graph

from gnn_utils import *
from actl import *

import numpy as np
import torch
import tqdm
import copy
import json
import shutil
import csv

seeds = np.array([12, 13, 16, 18, 21, 23, 29, 40, 50, 65])

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

with open("args.json", "r") as fp:
    args = json.load(fp)

print(args)

for i, s in enumerate(seeds):
    results_bacc = np.zeros_like(seeds, dtype=np.float32)

    pyg_graph_total = load_image_data(16)
    pyg_graph_train = load_image_data(10, hold_test=True)

    print("{}/{}".format(i+1,len(seeds)))

    args['random_state'] = s

    torch.manual_seed(s)
    np.random.seed(s)

    pyg_graph_train = get_train_val(pyg_graph_train, args)
    
    n = pyg_graph_train.num_nodes
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    pyg_graph_total = pyg_graph_total.to(device)
    pyg_graph_train = pyg_graph_train.to(device)
    
    model = getattr(arch.karate_graph, args["model"])(960, args['hidden_dim'], 2).to(device) 
    
    _, results_bacc[i] = run_base(model, pyg_graph_train, args, pyg_graph_total)

#print("Mean: {}".format(results_bacc.mean()))
#print("Std: {}".format(results_bacc.std()))
args["bacc_mean"] = results_bacc.mean()
args["bacc_std"] = results_bacc.std()

del args['random_state']

with open("../../../results/gnn.csv") as fp:
    r = csv.DictReader(fp)
    with open(".mycsvfile.csv", "w") as f2:
        w = csv.DictWriter(f2, args.keys())
        w.writeheader()
        for row in r:
            w.writerow(row)
        w.writerow(args)

shutil.move(".mycsvfile.csv", "../../../results/gnn.csv")

