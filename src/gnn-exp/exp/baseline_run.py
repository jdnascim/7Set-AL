"""
Run the GNN experiment for the full dataset
"""
import sys
sys.path.append("../")

import arch.gnn_arch

from gnn_utils import *
import numpy as np
import torch
import tqdm
import copy
import json
import csv
import shutil
import random
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--model", default=None, type=str, required=True,
                    help="model for the experiment")
parser.add_argument("--emb", default=None, type=str, required=True,
                    help="embeddings for the experiment")
parser.add_argument("--cuda_device", default=None, type=int, required=True,
                    help="gpu id")
parser.add_argument("--graph", default='knn', type=str, required=False,
                    help="graph")
parser.add_argument("--epochs", default=1000, type=int, required=False,
                    help="epochs")

args_cl = parser.parse_args()

seeds = np.array([12, 13, 16, 18, 21, 23, 29, 40, 50, 65])
results_bacc = np.zeros_like(seeds, dtype=np.float32)

with open("args.json", "r") as fp:
    args = json.load(fp)

for k in vars(args_cl):
    args[k] = getattr(args_cl, k)

app_csv = "_base"
for k in vars(args_cl):
    args[k] = getattr(args_cl, k)
    if k in ("model", "emb"):
        app_csv = app_csv + "_" + str(args[k])

results_file = args['results_file']

dev_id = args['cuda_device']
device = torch.device('cuda:{}'.format(dev_id) if torch.cuda.is_available() else 'cpu')

emb = args['emb']
graph = args['graph']

for i, s in enumerate(seeds):
    print("{}/{}".format(i+1,len(seeds)))

    args['random_state'] = s

    torch.manual_seed(s)
    np.random.seed(s)
    random.seed(s)

    pyg_graph_train = load_image_data(emb, graph, hold_test=True)
    pyg_graph_train = pyg_graph_train.to(device)
    
    inp_size = pyg_graph_train.x.shape[1]
    
    model = getattr(arch.karate_graph, args["model"])(inp_size, 2).to(device) 
    
    best_model, _ = run_base(model, pyg_graph_train, args)

    pyg_graph_total = load_image_data(emb, graph)
    pyg_graph_total = pyg_graph_total.to(device)

    results_bacc[i] = validate_best_model(model, pyg_graph_total, args)

args["bacc_mean"] = results_bacc.mean().round(4)
args["bacc_std"] = results_bacc.std().round(4)

del args['random_state']
del args['cuda_device']
del args['display']
del args['results_file']

args['train_size'] = int(pyg_graph_total.train_mask.sum())
args['val_size'] = int(pyg_graph_total.val_mask.sum())
args['test_size'] = int(pyg_graph_total.test_mask.sum())

with open(results_file.format(app_csv), "w") as f2:
    w = csv.DictWriter(f2, args.keys())
    w.writeheader()
    w.writerow(args)