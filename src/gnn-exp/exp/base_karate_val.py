import sys
sys.path.append("../")

import arch.karate_graph

from gnn_utils import *
import numpy as np
import torch
import tqdm
import copy
import json
import csv
import shutil

seeds = np.array([12, 13, 16, 18, 21, 23, 29, 40, 50, 65])
results_bacc = np.zeros_like(seeds, dtype=np.float32)

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

with open("args.json", "r") as fp:
    args = json.load(fp)

for i, s in enumerate(seeds):
    print("{}/{}".format(i+1,len(seeds)))

    args['random_state'] = s

    torch.manual_seed(s)
    np.random.seed(s)

    pyg_graph = load_image_data(16)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    pyg_graph = pyg_graph.to(device)
    
    model = getattr(arch.karate_graph, args["model"])(960, args['hidden_dim'], 2).to(device) 
    
    _, results_bacc[i] = run_base(model, pyg_graph, args)

args["bacc_mean"] = results_bacc.mean()
args["bacc_std"] = results_bacc.std()

with open("../../../results/gnn.csv") as fp:
    r = csv.DictReader(fp)
    with open(".mycsvfile.csv", "w") as f2:
        w = csv.DictWriter(f2, args.keys())
        w.writeheader()
        for row in r:
            w.writerow(row)
        w.writerow(args)

shutil.move(".mycsvfile.csv", "../../../results/gnn.csv")


    
