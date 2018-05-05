import numpy as np
from sklearn.base import BaseEstimator, ClusterMixin
import networkx as nx

import matplotlib.pyplot as plt

from sklearn.datasets import load_iris

iris = load_iris()

data = iris.data

x_t = 3


def _hill_climb(x_t, X, W=None, h=0.1, eps=1e-7):

    error = 99.
    prob = 0.
    x_l1 = np.copy(x_t)

    radius_new = 0.
    radius_old = 0.
    radius_twiceold = 0.
    iters = 0.
    while True:
        radius_thriceold = radius_twiceold
        radius_twiceold = radius_old
        radius_old = radius_new
        x_l0 = np.copy(x_l1)
        x_l1, density = _step(x_l0, X, W=W, h=h)
        error = density - prob
        prob = density
        radius_new = np.linalg.norm(x_l1 - x_l0)
        radius = radius_thriceold + radius_twiceold + radius_old + radius_new
        iters += 1
        if iters > 3 and error < eps:
            break
    return [x_l1, prob, radius]


def _step(x_l0, X, W=None, h=0.1):
    n = X.shape[0]
    d = X.shape[1]
    superweight = 0.
    x_l1 = np.zeros((1, d))
    if W is None:
        W = np.ones((n, 1))
    else:
        W = W
    for j in range(n):
        kernel = kernelize(x_l0, X[j], h, d)
        kernel = kernel * W[j] / (h ** d)
        superweight = superweight + kernel
        x_l1 = x_l1 + (kernel * X[j])
    x_l1 = x_l1 / superweight
    density = superweight / np.sum(W)
    return [x_l1, density]


def kernelize(x, y, h, degree):
    kernel = np.exp(-(np.linalg.norm(x - y) / h) ** 2. / 2.) / ((2. * np.pi) ** (degree / 2))
    return kernel


class DENCLUE(BaseEstimator, ClusterMixin):

    def __init__(self, h=None, eps=1e-4, min_density=0., metric='euclidean'):
        self.h = h
        self.eps = eps
        self.min_density = min_density
        self.metric = metric

    def fit(self, X, y=None, sample_weight=None):
        if not self.eps > 0.0:
            raise ValueError("eps must be positive.")
        self.n_samples = X.shape[0]
        self.n_features = X.shape[1]
        density_attractors = np.zeros((self.n_samples, self.n_features))
        radii = np.zeros((self.n_samples, 1))
        density = np.zeros((self.n_samples, 1))

        # create default values
        if self.h is None:
            self.h = np.std(X) / 5
        if sample_weight is None:
            sample_weight = np.ones((self.n_samples, 1))
        else:
            sample_weight = sample_weight

        # initialize all labels to noise
        labels = -np.ones(X.shape[0])

        # climb each hill
        for i in range(self.n_samples):
            density_attractors[i], density[i], radii[i] = _hill_climb(X[i], X, W=sample_weight,h=self.h, eps=self.eps)

        densitys = np.zeros(self.n_samples)
        for i in range(data.shape[0]):
            densitys[i] = obj.get_density(x=data[i], X=data)

        cluster_info = {}
        num_clusters = 0
        cluster_info[num_clusters] = {'instances': [0],
                                      'centroid': np.atleast_2d(density_attractors[0])}
        g_clusters = nx.Graph()
        for j1 in range(self.n_samples):
            g_clusters.add_node(j1, attr_dict={'attractor': density_attractors[j1], 'radius': radii[j1], 'density': densitys[j1]})


        for j1 in range(self.n_samples):
            for j2 in (x for x in range(self.n_samples) if x != j1):

                if g_clusters.has_edge(j1, j2):
                    continue
                diff = np.linalg.norm(g_clusters.node[j1]['attr_dict']['attractor'] - g_clusters.node[j2]['attr_dict']['attractor'])
                if diff <= (g_clusters.node[j1]['attr_dict']['radius'] + g_clusters.node[j1]['attr_dict']['radius']):
                    g_clusters.add_edge(j1, j2)


        clusters = list(nx.connected_component_subgraphs(g_clusters,copy=True))
        num_clusters = 0


        for clust in clusters:


            max_instance = max(clust, key=lambda x: clust.node[x]['attr_dict']['density'])
            max_density = clust.node[max_instance]['attr_dict']['density']
            max_centroid = clust.node[max_instance]['attr_dict']['attractor']


            complete = False
            c_size = len(clust.nodes())
            if clust.number_of_edges() == (c_size * (c_size - 1)) / 2.:
                complete = True
            cluster_info[num_clusters] = {'instances': clust.nodes(),
                                          'size': c_size,
                                          'centroid': max_centroid,
                                          'density': max_density,
                                          'complete': complete}


            if max_density >= self.min_density:
                labels[clust.nodes()] = num_clusters
            num_clusters += 1

        self.clust_info_ = cluster_info
        self.labels_ = labels
        nx.draw(g_clusters)
        plt.show()
        return self

    def get_density(self, x, X, y=None, sample_weight=None):
        superweight = 0.
        n_samples = X.shape[0]
        n_features = X.shape[1]
        if sample_weight is None:
            sample_weight = np.ones((n_samples, 1))
        else:
            sample_weight = sample_weight
        for y in range(n_samples):
            kernel = kernelize(x, X[y], h=self.h, degree=n_features)
            kernel = kernel * sample_weight[y] / (self.h ** n_features)
            superweight = superweight + kernel
        density = superweight / np.sum(sample_weight)
        return density


obj = DENCLUE(h=0.1, eps=0.0001, min_density=2.85)

obj.fit(data)





