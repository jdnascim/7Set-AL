BASE = "/home/jnascimento/exps/2022-7set-al/7Set-AL/"

import sys
sys.path.append(BASE + "src/gnn-exp/")

import arch.gnn_arch

from gnn_utils import *

import numpy as np
import torch
import tqdm
import copy
import json
import shutil
import csv
import random
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--model", default=None, type=str, required=True,
                    help="model for the experiment")
parser.add_argument("--emb", default=None, type=str, required=True,
                    help="embeddings for the experiment")
parser.add_argument("--actl", default=None, type=str, required=True,
                    help="active learning method for the experiment")
parser.add_argument("--train_size", default=None, type=int, required=True,
                    help="training size for the exp")
parser.add_argument("--val_size", default=None, type=int, required=True,
                    help="val size for the exp")
parser.add_argument("--cuda_device", default=None, type=int, required=True,
                    help="gpu id")
parser.add_argument("--graph", default='knn', type=str, required=False,
                    help="graph")
parser.add_argument("--epochs", default=1000, type=int, required=False,
                    help="epochs")

args_cl = parser.parse_args()

with open("args.json", "r") as fp:
    args = json.load(fp)

app_csv = ""
for k in vars(args_cl):
    args[k] = getattr(args_cl, k)
    if k in ("model", "emb", "actl", "train_size"):
        app_csv = app_csv + "_" + str(args[k])
    
dev_id = args['cuda_device']
device = torch.device('cuda:{}'.format(dev_id) if torch.cuda.is_available() else 'cpu')

emb = args['emb']
graph = args['graph']
results_file  = args['results_file'] + ""

seeds = np.array([12, 13, 16, 18, 21, 23, 29, 40, 50, 65])
results_bacc = np.zeros_like(seeds, dtype=np.float32)

for i, s in enumerate(seeds):

    print("{}/{}".format(i+1,len(seeds)))

    args['random_state'] = s

    torch.manual_seed(s)
    np.random.seed(s)
    random.seed(s)

    pyg_graph_train = load_image_data(emb, graph, hold_test=True)
    pyg_graph_train = get_train_val(pyg_graph_train, args)
    pyg_graph_train = pyg_graph_train.to(device)

    n = pyg_graph_train.num_nodes
    
    inp_size = pyg_graph_train.x.shape[1]
    
    model = getattr(arch.karate_graph, args["model"])(inp_size, 2).to(device) 
    
    best_model, _ = run_base(model, pyg_graph_train, args)

    pyg_graph_total = load_image_data(emb, graph)
    pyg_graph_total = pyg_graph_total.to(device)

    results_bacc[i] = validate_best_model(model, pyg_graph_total, args)

args["bacc_mean"] = results_bacc.mean().round(4)
args["bacc_std"] = results_bacc.std().round(4)

del args['random_state']
del args['display']
del args['cuda_device']
del args['results_file']

#r = csv.DictReader(fp)
with open(results_file.format(app_csv), "w") as f2:
    w = csv.DictWriter(f2, args.keys())
    w.writeheader()
#    for row in r:
#        w.writerow(row)
    w.writerow(args)