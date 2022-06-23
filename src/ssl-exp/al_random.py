BASE = "/home/jnascimento/exps/2022-7set-al/7Set-AL/"

import sys
sys.path.append(BASE + "src")

from sklearn.semi_supervised import LabelSpreading
from utils import get_emb_vec, get_normalized_acc
from tqdm import tqdm
import numpy as np

QTDE_EXP = 10
TRAIN_ANNO_SIZE = 30

np.random.seed(13)

out = get_emb_vec('clipcat')

train_mask = out["train_mask"] 
eval_mask = out["eval_mask"]
test_mask = out["test_mask"]

bacc_list = np.zeros([QTDE_EXP], dtype=np.float32)

for i in tqdm(range(QTDE_EXP)):
    X_train = out["emb_mt"][train_mask | eval_mask]
    y_train = out["anno"][train_mask | eval_mask].copy()

    N_train = X_train.shape[0]

    train_anno_ix = np.random.choice(range(N_train), TRAIN_ANNO_SIZE,
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
    
    bacc_list[i] = get_normalized_acc(y_test, y_pred)
    
print("BAcc - Mean: {:.4f}".format(bacc_list.mean()))
print("BAcc - Std: {:.4f}".format(bacc_list.std()))