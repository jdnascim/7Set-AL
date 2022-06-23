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
import random


seeds = np.array([12, 13, 16, 18, 21, 23, 29, 40, 50, 65])
results_bacc = np.zeros_like(seeds, dtype=np.float32)

with open("args.json", "r") as fp:
    args = json.load(fp)

result_file = args['result_file']

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

    pyg_graph = load_image_data(emb, graph)
    
    pyg_graph = pyg_graph.to(device)

    inp_size = pyg_graph.x.shape[1]
    
    model = getattr(arch.karate_graph, args["model"])(inp_size, args['hidden_dim'], 2).to(device) 
    
    _, results_bacc[i] = run_base(model, pyg_graph, args)

args["bacc_mean"] = results_bacc.mean().round(4)
args["bacc_std"] = results_bacc.std().round(4)

del args['random_state']
del args['cuda_device']
del args['display']

with open(RESULTS_FILE) as fp:
    r = csv.DictReader(fp)
    with open(".mycsvfile.csv", "w") as f2:
        w = csv.DictWriter(f2, args.keys())
        w.writeheader()
        for row in r:
            w.writerow(row)
        w.writerow(args)

shutil.move(".mycsvfile.csv", RESULTS_FILE)