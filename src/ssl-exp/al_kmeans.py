BASE = "/home/jnascimento/exps/2022-7set-al/7Set-AL/"

import sys
sys.path.append(BASE + "src")

from sklearn.semi_supervised import LabelSpreading
from sklearn.cluster import KMeans
from utils import get_emb_vec, get_normalized_acc
from tqdm import tqdm
import numpy as np

QTDE_EXP = 10
TRAIN_ANNO_SIZE = 30

np.random.seed(13)

out = get_emb_vec('clipsum')

train_mask = out["train_mask"] 
eval_mask = out["eval_mask"]
test_mask = out["test_mask"]

bacc_list = np.zeros([QTDE_EXP], dtype=np.float32)

for i in tqdm(range(QTDE_EXP)):
    X_train = out["emb_mt"][train_mask | eval_mask]
    y_train = out["anno"][train_mask | eval_mask].copy()

    N_train = X_train.shape[0]

    model_kmeans = KMeans(n_clusters=TRAIN_ANNO_SIZE,
                          random_state=i).fit(X_train)
    
    dist = model_kmeans.transform(X_train)  # (X.shape[0], qtde)

    # indices ordered by value. ord_argdist shape: (X.shape[0] * qtde, 2)
    ord_argdist = np.dstack(np.unravel_index(np.argsort(dist.ravel()),
                                             (dist.shape[0],
                                              dist.shape[1])))[0] 

    sample_ix = np.full(TRAIN_ANNO_SIZE, -1, dtype=np.int32)

    # find the closest item to each cluster
    missing_samples = TRAIN_ANNO_SIZE
    ix = 0
    while missing_samples > 0:
        if sample_ix[ord_argdist[ix][1]] == -1:
            sample_ix[ord_argdist[ix][1]] = ord_argdist[ix][0]
            missing_samples -= 1
        ix += 1
    
    train_unlbl_mask = np.ones(N_train, np.bool_)
    train_unlbl_mask[sample_ix] = False

    # set unlabelled items as -1
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