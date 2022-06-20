import numpy as np
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


def get_train_val(pyg_graph, args):
    actl = args["actl"]

    if actl == "random":
        return select_random(pyg_graph, args)
    elif actl == "degree":
        return select_high_degree(pyg_graph, args)
    elif actl == "kmeans":
        return select_kmeans_centers(pyg_graph, args)