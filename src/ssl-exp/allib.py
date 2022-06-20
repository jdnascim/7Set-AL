import sys
sys.path.append("../")

from abc import abstractmethod
from sklearn.semi_supervised import LabelSpreading
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import cosine_distances
from utils import get_normalized_acc, get_f1
import numpy as np


class albase:
    def __init__(self, qtde_exp, lbl_qtde, random_seed=13):
        """ Base class for AL methods

        Args:
            qtde_exp (_type_): amount of times that a same experiment
            will be performed to measure mean and std deviation

            lbl_qtde (_type_): vector consisting the amount of labels
            in each iteration of the al method

            random_seed (int, optional): seed. Defaults to 13.
        """

        self.qtde_exp = qtde_exp
        self.lbl_qtde = lbl_qtde

        np.random.seed(random_seed)

        # init bacc and f1 variables - where the results are going
        # to be stored
        self.bacc = np.zeros([len(self.lbl_qtde), self.qtde_exp],
                             dtype=np.float32)
        self.f1 = np.zeros([len(self.lbl_qtde), self.qtde_exp],
                           dtype=np.float32)
    

    def ssl(self, X, lbl, train_ix, unlabelled_ix):
        """ Performs semi-supervised learning

        Args:
            X (_type_): feature vector (N, D)
            lbl (_type_): label vector (N,)
            train_ix (_type_): train set indexes (T, )

        Returns:
            bacc (_float_): balanced accuracy
            f1 (_float_): f1
            model_lgc (LabelSpreading): trained model
        """

        # unlabelled_ix = np.array([i for i in np.arange(len(lbl)) if i not in
        #                          set(train_ix)])
        X = X[np.concatenate((train_ix, unlabelled_ix))]

        y_train = lbl.copy()
        y_train[unlabelled_ix] = -1
        y_train = y_train[np.concatenate((train_ix, unlabelled_ix))]
        
        model_lgc = LabelSpreading(kernel='knn', alpha=0.5, max_iter=300,
                                   n_neighbors=16, n_jobs=-1).fit(X, y_train)

        y_test = lbl[unlabelled_ix]

        # prediction
        y_res = model_lgc.transduction_[len(train_ix):]

        # evaluate prediction
        bacc = get_normalized_acc(y_test, y_res)
        f1 = get_f1(y_test, y_res)

        return bacc, f1, model_lgc

    @abstractmethod
    def select(self, X, qtde):
        """ select items of X. This function should be implemented for any
        AL method that is created, since it represents the core of that

        Args:
            X (_type_): feature matrix
            qtde (_type_): qtde of items that is going ot be sampled
        """
        pass

    def sampling_dataset(self, X, sample=None, sample_size=-1):
        """

        Args:
            X (_type_): _description_
            sample (_type_, optional): _description_. Defaults to None.
            sample_size (int, optional): _description_. Defaults to -1.
        """

        if sample == "random" and sample_size != -1:
            sample_ix = np.random.choice(np.arange(X.shape[0]), sample_size,
                                         replace=False)
        elif sample == "kmeans" and sample_size != -1:
            model_kmeans = KMeans(n_clusters=sample_size,
                                  n_init=1).fit(X)
            dist = model_kmeans.transform(X)

            ord_argdist = np.dstack(np.unravel_index(np.argsort(dist.ravel()),
                                                     (10, 20)))[0]

            sample_ix = np.full(sample_size, -1, dtype=np.int32)

            missing_samples = sample_size
            ix = 0
            while missing_samples > 0:
                if sample_ix[ord_argdist[ix][1]] == -1:
                    sample_ix[ord_argdist[ix][1]] = ord_argdist[ix][0]
                    missing_samples -= 1

            sample_ix = np.argmin(dist, axis=0)
        else:
            # no sampling - X[sample_ix] = X
            sample_ix = np.arange(X.shape[0])

        if sample is not None and sample_size != -1:
            unsampled_ix = np.array([i for i in np.arange(X.shape[0])
                                     if i not in np.unique(sample_ix)])
        else:
            unsampled_ix = np.arange(X.shape[0])

        ix_sampled = dict()
        for ix, smp in enumerate(sample_ix):
            ix_sampled[smp] = ix

        ix_unsampled = dict()
        for ix, smp in enumerate(unsampled_ix):
            ix_unsampled[smp] = ix
        
        # if sample_size = -1 or sample = none, sample = unsumpled
        # sample_ix and unsampled_ix = indexes
        # ix_sampled and ix_unsampled: dict: key = true ix, val = smp ix 
        self.sample_ix = sample_ix
        self.ix_sampled = ix_sampled
        self.unsampled_ix = unsampled_ix
        self.ix_unsampled = ix_unsampled
    
    def __itera(self, X_test, train_ix_sampled, itera, model, qtde):
        pred = model.predict(X_test)
        proba = model.predict_proba(X_test)

        certainty = proba[(np.arange(len(pred)), pred)]

        if itera == "unc":
            most_uncertain = np.argsort(certainty)
            k = 0
            while len(np.unique(train_ix_sampled)) < qtde:
                if most_uncertain[k] not in np.unique(train_ix_sampled):
                    train_ix_sampled = np.append(train_ix_sampled,
                                                 most_uncertain[k])
                k += 1

        elif itera == "c":
            most_certain = np.argsort(certainty)[::-1]
            k = 0
            while len(np.unique(train_ix_sampled)) < qtde:
                if most_certain[k] not in np.unique(train_ix_sampled):
                    train_ix_sampled = np.append(train_ix_sampled,
                                                 most_certain[k])
                k += 1
        else:
            raise("\"itera\" should be \"unc\" or \"c\".")

        return train_ix_sampled

    def sample_ssl(self, X, lbl, sample=None, sample_size=-1, itera='unc',
                   display=False):
        """_summary_

        Args:
            X (_type_): Feature Matrix
            lbl (_type_): Label Vector
            sample (_type_, optional): Method of sampling. Defaults to None.
            sample_size (int, optional): Size of sampling. Defaults to -1.
            iter (str, optional): Iterative method. Defaults to 'unc'.

        Returns:
            _type_: _description_
        """

        # first step: sampling - check implementation to see which
        # new attributes the object will have
        self.sampling_dataset(X, sample, sample_size)

        # view of X consiting the sampled matrix. Sampled matrix in this
        # case is the matrix that the items will be evaluated so they
        # can potentially be part of the training set
        X_sampled = X[self.sample_ix]

        for j in range(self.qtde_exp):
            print("{} of {}".format(j+1, self.qtde_exp))

            model = None
            train_ix_sampled = None
            train_ix = None

            # only goes to iteration process if there are 
            # more than one label on the training set
            diverse = False

            for i, qtde in enumerate((self.lbl_qtde)):
                print("{} of {}".format(i+1, len(self.lbl_qtde)))

                # if the labels are not diverse, can't use iterative 
                # methods unc or c
                if i == 0 or itera is None or diverse is False:
                    # new labelled items from the previously-sampled dataset
                    # indexes on the X_sampled
                    train_ix_sampled = self.select(X_sampled, qtde)

                else:
                    train_ix_sampled = self.__itera(X_sampled,
                                                    train_ix_sampled,
                                                    itera, model, qtde)

                # get indexes of the new labelled items in the main matrix
                train_ix = self.sample_ix[train_ix_sampled]

                # indexes on the main matrix of the sampled items that 
                # are not part of train_ix
                val_ix = np.array([smp for smp in self.sample_ix
                                   if smp not in np.unique(train_ix)])

                # assess the main items
                bacc_, f1_, model = self.ssl(X, lbl, train_ix, val_ix)

                if sample is not None:
                    # test_ix is the whole (unsampled) items that are[
                    # not part of train_ix
                    test_ix = [smp for smp in self.unsampled_ix
                               if smp not in set(train_ix)]
                    
                    y_res = model.predict(X[test_ix])
                    y_test = lbl[test_ix] 
        
                    # evaluate prediction
                    bacc_ = get_normalized_acc(y_test, y_res)
                    f1_ = get_f1(y_test, y_res)

                self.bacc[i][j] = bacc_
                self.f1[i][j] = f1_

                # bacc_, f1_, model = self.ssl(X, lbl, train_ix, test_ix)
                # self.bacc[i][j] = bacc_
                # self.f1[i][j] = f1_

                # check if the labels from the training set are diverse
                if len(np.unique(lbl[train_ix])) > 1:
                    diverse = True

        return self.results(display)

    def results(self, display=False):
        if display is True:
            bacc_mean = "BAcc - Mean: {}".format(np.mean(self.bacc, axis=1))
            bacc_std = "BAcc - Std: {}".format(np.std(self.bacc, axis=1))
            f1_mean = "F1 - Mean: {}".format(np.mean(self.f1, axis=1))
            f1_std = "F1 - Std: {}".format(np.std(self.f1, axis=1))

            print(bacc_mean)
            print(bacc_std)
            print(f1_mean)
            print(f1_std)
        else:
            bacc_mean = np.mean(self.bacc, axis=1)
            bacc_std = np.std(self.bacc, axis=1)
            f1_mean = np.mean(self.f1, axis=1)
            f1_std = np.std(self.f1, axis=1)

        return bacc_mean, bacc_std, f1_mean, f1_std


class random_sampling(albase):
    def __init__(self, qtde_exp, lbl_qtde, random_seed=13):
        albase.__init__(self, qtde_exp, lbl_qtde, random_seed)

    def select(self, X, qtde, sample=None):
        return np.random.choice(range(X.shape[0]), qtde,
                                replace=False)


class kmeans_sampling(albase):
    def __init__(self, qtde_exp, lbl_qtde, random_seed=13):
        albase.__init__(self, qtde_exp, lbl_qtde, random_seed)

        self.random_seed = random_seed

    def select(self, X, qtde):
        model_kmeans = KMeans(n_clusters=qtde,
                              random_state=self.random_seed).fit(X)
        
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

        return sample_ix


class density(albase):
    def __init__(self, qtde_exp, lbl_qtde, distance, random_seed=13):
        self.distance = distance
        qtde_exp = 1
        albase.__init__(self, qtde_exp, lbl_qtde, random_seed)

    def select(self, X, qtde, sample=None):
        if self.distance == 'rbf':
            tmpdist = -0.5 * euclidean_distances(X) ** 2
            lnC = -1 * np.max(tmpdist)
            dist = np.exp(tmpdist + lnC) / np.exp(lnC)
        elif self.distance == 'euclidean':
            dist = euclidean_distances(X)
        elif self.distance == 'cosine':
            dist = cosine_distances(X)

        repre = dist.sum(axis=0) / (len(X) - 1)

        return np.argpartition(repre, -1 * qtde)[-1 * qtde:]


class rbf(density):
    def __init__(self, qtde_exp, lbl_qtde, random_seed=13):
        density.__init__(self, qtde_exp, lbl_qtde, "rbf", random_seed)


class euclidean(density):
    def __init__(self, qtde_exp, lbl_qtde, random_seed=13):
        density.__init__(self, qtde_exp, lbl_qtde, "euclidean", random_seed)


class cosine(density):
    def __init__(self, qtde_exp, lbl_qtde, random_seed=13):
        density.__init__(self, qtde_exp, lbl_qtde, "cosine", random_seed)
