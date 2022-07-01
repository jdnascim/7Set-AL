BASE = "/home/jnascimento/exps/2022-7set-al/7Set-AL/"

import sys
sys.path.append(BASE + "src")

from sklearn.semi_supervised import LabelSpreading
from utils import get_emb_vec, get_normalized_acc
from tqdm import tqdm
import numpy as np
import argparse
import csv
import shutil

parser = argparse.ArgumentParser()

parser.add_argument("--emb", default=None, type=str, required=True,
                    help="embeddings for the experiment")
parser.add_argument("--train_size", default=None, type=int, required=True,
                    help="training size for the exp")

args_cl = parser.parse_args()

args = dict()
for k in vars(args_cl):
    args[k] = getattr(args_cl, k)

args["actl"] = "random"

QTDE_EXP = 10

np.random.seed(13)

emb = args["emb"]
train_size = args["train_size"]

out = get_emb_vec(emb)

train_mask = out["train_mask"] 
eval_mask = out["eval_mask"]
test_mask = out["test_mask"]

results_bacc = np.zeros([QTDE_EXP], dtype=np.float32)

for i in tqdm(range(QTDE_EXP)):
    X_train = out["emb_mt"][train_mask | eval_mask]
    y_train = out["anno"][train_mask | eval_mask].copy()

    N_train = X_train.shape[0]

    train_anno_ix = np.random.choice(range(N_train), train_size,
                                     replace=False)
    
    train_unlbl_mask = np.ones(N_train, np.bool_)
    train_unlbl_mask[train_anno_ix] = False
    train_anno_mask = np.zeros(N_train, np.bool_)
    train_anno_mask[train_anno_ix] = True

    y_train[train_unlbl_mask] = -1
    
    model_lgc = LabelSpreading(kernel='knn', alpha=0.5, max_iter=300,
                            n_neighbors=16, n_jobs=-1)
    model_lgc.fit(X_train, y_train)
    
    X_test = out['emb_mt'][test_mask]
    y_test = out['anno'][test_mask]
    
    y_pred = model_lgc.predict(X_test)
    
    results_bacc[i] = get_normalized_acc(y_test, y_pred)
    
args["bacc_mean"] = results_bacc.mean().round(4)
args["bacc_std"] = results_bacc.std().round(4)

with open(BASE + "results/results_lgc.csv", "r") as fp:
    r = csv.DictReader(fp)
    with open(".mycsv.csv", "w") as f2:
        w = csv.DictWriter(f2, args.keys())
        w.writeheader()
        for row in r:
            w.writerow(row)
        w.writerow(args)
shutil.move(".mycsv.csv", BASE + "results/results_lgc.csv")