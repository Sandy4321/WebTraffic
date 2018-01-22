import numpy as np

from tslearn.piecewise import SymbolicAggregateApproximation
from sklearn.cluster import DBSCAN
from sklearn.cluster import SpectralClustering
from sklearn.cluster import AgglomerativeClustering


def sax_sim_matrix(df: np.ndarray, word_len, alphabet_len):
    '''
    Computes the sax distance for a series,
    with specified alphabet length and word length
    '''
    sax = SymbolicAggregateApproximation(word_len, alphabet_len)
    sax.fit(df)

    n_series = df.shape[0]
    sim_matrix = np.zeros((n_series, n_series))

    for i in range(n_series):
        for j in range(n_series):
            sim_matrix[i][j] = sax.distance(df[i], df[j])

    return sim_matrix


def dbscan(sim_matrix):
    '''
    Clusters timeseries by sax fingerprint similarity
    '''
    dbscan = DBSCAN(metric='precomputed', n_jobs=-1)

    clusters = dbscan.fit_predict(sim_matrix)
    return clusters


def spectral(sim_matrix, n_clusters=4):
    '''
    Cluster tseries by spectral clustering
    '''
    spectral = SpectralClustering(n_clusters=n_clusters, affinity='precomputed', n_jobs=-1)
    clusters = spectral.fit_predict(sim_matrix)
    return clusters


def hierarchichal(sim_matrix):
    '''
    Cluster tseries by hierarchichal aggregation
    '''
    hierarchichal = AgglomerativeClustering(n_clusters=4, linkage='complete', affinity='precomputed')
    clusters = hierarchichal.fit_predict(sim_matrix)
    return clusters
