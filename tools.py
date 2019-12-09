from sklearn.manifold import TSNE
import numpy as np

from itertools import permutations
import scipy


def eval_kendall_tau(y):
    taus = []
    for s, t in permutations(range(len(y)), 2):
        u = np.array(y[s])
        v = np.array(y[t])
        dists = scipy.spatial.distance.cdist(u, v, 'sqeuclidean')
        nns = np.argmin(dists, axis=1)
        taus.append(scipy.stats.kendalltau(
            np.arange(len(nns)), nns).correlation)
    taus = np.array(taus)
    taus = taus[~np.isnan(taus)]
    tau = np.mean(taus)
    return tau


def eval_tsne(y, num):
    label = []
    y_flat = []
    for i, y_tmp in enumerate(y[:num]):
        y_tmp = y_tmp.reshape(-1, 128)
        for j in range(len(y_tmp)):
            y_flat.append(y_tmp[j, :])
        label += [i for i in range(len(y_tmp))]

    model = TSNE(n_components=2, init='random', random_state=0, perplexity=10)
    Y = model.fit_transform(y_flat)

    return Y
