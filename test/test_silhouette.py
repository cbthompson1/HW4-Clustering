import pytest
import numpy as np
from cluster import kmeans, silhouette, utils
from sklearn.metrics import silhouette_score

def test_bad_silhouette_input():
    """Confirm mismatched observations and labels throws an error.ß"""
    mat, _ = utils.make_clusters(n=100)
    _, truth = utils.make_clusters(n=101)
    sil = silhouette.Silhouette()
    with pytest.raises(ValueError):
        sil.score(mat, truth)

def test_silhouette_score():
    """
    Same test as in kmeans_test (modifying scale), but check the silouette
    score against the sklearn equivalent to ensure completeness. Because we are
    dealing with floats, check the first 10 digits are good.
    """
    scale = [0.01, 0.1, 1, 3]
    for factor in scale:
        n = 1000
        m = 2
        k = 3
        model = kmeans.KMeans(k=3, dims=2)
        mat, _ = utils.make_clusters(n=n, m=m, k=k, scale=factor, seed=161803)
        model.fit(mat)
        pred = model.predict(mat)
        sil = silhouette.Silhouette()
        silhouette_score_inhouse = round(np.mean(sil.score(mat, pred)), 10)
        sklearn_score = round(silhouette_score(mat, pred), 10)
        assert sklearn_score == silhouette_score_inhouse
