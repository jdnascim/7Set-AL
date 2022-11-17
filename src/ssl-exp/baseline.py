"""
LGC experiment
"""

BASE = "/home/jnascimento/exps/2022-7set-al/7Set-AL/"

import sys
sys.path.append(BASE + "src")

from sklearn.semi_supervised import LabelSpreading
from utils import get_emb_vec, get_normalized_acc
import numpy as np
import itertools

out = get_emb_vec('clipsum')

train_mask = out["train_mask"] 
eval_mask = out["eval_mask"]
test_mask = out["test_mask"]

X_train = out["emb_mt"][train_mask | eval_mask]

tmp_y = out["anno"].copy()
tmp_y[eval_mask] = -1
y_train = tmp_y[train_mask | eval_mask]

mask_unlabelled = np.array(y_train == -1, dtype=np.bool_)

model_lgc = LabelSpreading(kernel='knn', alpha=0.5, max_iter=300,
                           n_neighbors=16, n_jobs=-1)
model_lgc.fit(X_train, y_train)

X_test = out['emb_mt'][test_mask]
y_test = out['anno'][test_mask]

y_pred = model_lgc.predict(X_test)

bacc = get_normalized_acc(y_test, y_pred)

print(bacc)