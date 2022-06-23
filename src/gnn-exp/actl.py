BASE = "/home/jnascimento/exps/2022-7set-al/7Set-AL/"

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from torch_geometric.utils import degree
from sklearn.model_selection import train_test_split

def select_kmeans_centers(pyg_graph, args):
    X = pyg_graph.x
    n = pyg_graph.num_nodes
    y = pyg_graph.y

    train_size = args['train_size']
    val_size = args['val_size']
    seed = args['random_state']

    qtde = train_size + val_size


    model_kmeans = KMeans(n_clusters=qtde,
                          random_state=seed).fit(X)
    
    dist = model_kmeans.transform(X)  # (X.shape[0], qtde)

    # indices ordered by value. ord_argdist shape: (X.shape[0] * qtde, 2)
    ord_argdist = np.dstack(np.unravel_index(np.argsort(dist.ravel()),
                                             (dist.shape[0],
                                              dist.shape[1])))[0] 

    sample_ix = np.full(qtde, -1, dtype=np.int32)

    # find the closest item to each cluster
    missing_samples = qtde
    ix = 0
    while missing_samples > 0:
        if sample_ix[ord_argdist[ix][1]] == -1:
            sample_ix[ord_argdist[ix][1]] = ord_argdist[ix][0]
            missing_samples -= 1
        ix += 1

    train_ix, val_ix, _, _ = train_test_split(sample_ix, y[sample_ix], train_size=train_size, stratify=y[sample_ix], random_state=seed)
    
    pyg_graph.train_mask = np.zeros(n, np.bool_)
    pyg_graph.train_mask[train_ix] = True
    pyg_graph.val_mask = np.zeros(n, np.bool_)
    pyg_graph.val_mask[val_ix] = True
    pyg_graph.test_mask = np.ones(n, np.bool_)
    pyg_graph.test_mask[sample_ix] = False

    return pyg_graph


def select_high_degree(pyg_graph, args):
    # training set
    n = pyg_graph.num_nodes

    train_size = args['train_size']
    val_size = args['val_size']

    deg = degree(pyg_graph.edge_index[1], n)
    partition_nodes = np.argpartition(deg, -1 * train_size)
    train_ix = partition_nodes[-1 * train_size:]
    
    pyg_graph.train_mask = np.zeros(n, np.bool_)
    pyg_graph.train_mask[train_ix] = True
    
    partition_nodes = np.argpartition(deg, -1 * (train_size + val_size))
    val_ix = partition_nodes[-1 * (train_size + val_size):-1 * train_size]
    
    pyg_graph.val_mask = np.zeros(n, np.bool_)
    pyg_graph.val_mask[val_ix] = True
    
    # test set
    pyg_graph.test_mask = np.ones(n, np.bool_)
    pyg_graph.test_mask[train_ix] = False
    pyg_graph.test_mask[val_ix] = False
    
    return pyg_graph


def select_random(pyg_graph, args):
    train = args['train_size']
    val = args['val_size']
    test = args['test_size']
    s = args['random_state']

    np.random.seed(s)

    n = pyg_graph.num_nodes

    if test == -1:
        test = n - train - val
    
    val_test_ix = np.random.choice(np.arange(n), n-train, replace=False)
    
    pyg_graph.train_mask = np.ones(n, np.bool_)
    pyg_graph.train_mask[val_test_ix] = False

    if val != 0 and test != 0:
        val_ix = np.random.choice(val_test_ix, val, replace=False)
        pyg_graph.val_mask = np.zeros(n, np.bool_)
        pyg_graph.val_mask[val_ix] = True

        pyg_graph.test_mask = np.ones(n, np.bool_)
        pyg_graph.test_mask[pyg_graph.train_mask | pyg_graph.val_mask] = False
    elif val == 0:
        pyg_graph.test_mask = np.zeros(n, np.bool_)
        pyg_graph.test_mask[val_test_ix] = True
    elif test == 0:
        pyg_graph.val_mask = np.zeros(n, np.bool_)
        pyg_graph.val_mask[val_test_ix] = True

    return pyg_graph


def select_clustering(pyg_graph, args, filename):
    np.random.seed(args['random_seed'])

    df_repr = pd.read_csv(BASE + "clusterings/" + filename)

    train_size = args['train_size']
    val_size = args['val_size']

    n_clusters = df_repr.cluster_id.unique().shape[0]

    if (train_size + val_size) % n_clusters != 0:
        raise("train + val not multiple of {}".format(n_clusters))
    
    train_val_set = set()

    items_per_cluster = (train_size + val_size) // n_clusters

    for i in range(n_clusters):
        repr_cl = set(df_repr[df_repr.cluster_id == i][:items_per_cluster].tweet_id)
        train_val_set.update(repr_cl)
    
    print(train_val_set)

    n = pyg_graph.num_nodes

    train_val_mask = np.zeros(n, np.bool_)

    train_val_ix = set()

    for i in range(n):
        if pyg_graph.tweet_id[i] in train_val_set:
            train_val_ix.add(i)
    
    train_val_ix = np.array(list(train_val_ix))

    train_ix = np.random.choice(train_val_ix, train_size,
                         replace=False)

    pyg_graph.train_mask = np.zeros(n, np.bool_)
    pyg_graph.train_mask[train_ix] = True

    pyg_graph.val_mask = np.zeros(n, np.bool_)
    pyg_graph.val_mask[train_val_ix] = True
    pyg_graph.val_mask[train_ix] = False

    pyg_graph.test_mask = np.ones(n, np.bool_)
    pyg_graph.test_mask[train_val_ix] = False

    return pyg_graph


def get_train_val(pyg_graph, args):
    actl = args["actl"]
    
    if args["emb"] == "clipcat":
        suffix = "Cat"
    elif args['emb'] == "clipsum":
        suffix = "Sum"

    clustering_ref = {
        "leiden-bw":"leiden/LeidenBetweenness{}.csv".format(suffix),
        "leiden-close":"leiden/LeidenCloseness{}.csv".format(suffix),
        "leiden-deg":"leiden/LeidenDegree{}.csv".format(suffix),
        "leiden-eigen":"leiden/LeidenEigen{}.csv".format(suffix),
        "leiden-mci":"leiden/LeidenMCI{}.csv".format(suffix),
        "leiden-pgrk":"leiden/LeidenPageRank{}.csv".format(suffix),
        "leiden-rdn":"leiden/LeidenRandom{}.csv".format(suffix),
        "agg-bw":"agglomerative/AggBetweenness{}.csv".format(suffix),
        "agg-close":"agglomerative/AggCloseness{}.csv".format(suffix),
        "agg-deg":"agglomerative/AggDegree{}.csv".format(suffix),
        "agg-eigen":"agglomerative/AggEigen{}.csv".format(suffix),
        "agg-mci":"agglomerative/AggMCI{}.csv".format(suffix),
        "agg-pgrk":"agglomerative/AggPageRank{}.csv".format(suffix),
        "agg-rdn":"agglomerative/AggRandom{}.csv".format(suffix)
    }

    if actl == "random":
        return select_random(pyg_graph, args)
    elif actl == "degree":
        return select_high_degree(pyg_graph, args)
    elif actl == "kmeans":
        return select_kmeans_centers(pyg_graph, args)
    elif actl in clustering_ref.keys():
        return select_clustering(pyg_graph, args, clustering_ref[actl])