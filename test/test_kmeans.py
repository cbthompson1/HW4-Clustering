import pytest
import numpy as np
from cluster import kmeans, silhouette, utils


# Init test methods.
def test_init_max_iter_negative():
    """Model should raise assert for negative iterations."""
    with pytest.raises(ValueError):
        _ = kmeans.KMeans(k=3, dims=3, max_iter=-1)

def test_init_max_iter_zero():
    """Model should raise assert for 0 iterations (implementation choice)."""
    with pytest.raises(ValueError):
        _ = kmeans.KMeans(k=3, dims=3, max_iter=0)

def test_init_negative_dims():
    """Model should raise assert for negative dim values."""
    with pytest.raises(ValueError):
        _ = kmeans.KMeans(k=3, dims=-1)

def test_init_zero_dims():
    """Model should raise assert for zero dim values."""
    with pytest.raises(ValueError):
        _ = kmeans.KMeans(k=3, dims=0)

def test_init_negative_k():
    """Model should raise assert for negative k values."""
    with pytest.raises(ValueError):
        _ = kmeans.KMeans(k=-1, dims=2)

def test_init_zero_k():
    """Model should raise assert for k == 0."""
    with pytest.raises(ValueError):
        _ = kmeans.KMeans(k=0, dims=2)

def test_init_k_greater_than_n():
    """Model should raise assert when clusters outnumbers observations."""
    model = kmeans.KMeans(k=300, dims=2)
    mat, _ = utils.make_clusters(n=299, m=2, k=2)
    with pytest.raises(ValueError):
        model.fit(mat)

# Get error tests - though calculable, just go off vibes.
def test_get_error_trivial():
    """Test error function returns 0 when k = n."""
    model = kmeans.KMeans(k=3, dims=2)
    mat, _ = utils.make_clusters(n=3, m=2, k=3)
    model.fit(mat)
    assert model.get_error() == 0

def test_get_error_low_n():
    """Model should have extremely low error with only one additional point."""
    model = kmeans.KMeans(k=3, dims=2)
    mat, _ = utils.make_clusters(n=4, m=2, k=3)
    model.fit(mat)
    assert model.get_error() < 0.1

def test_get_error_many_points():
    """Model with many points should have moderate error."""
    model = kmeans.KMeans(k=3, dims=2)
    mat, _ = utils.make_clusters(n=100, m=2, k=3)
    model.fit(mat)
    assert model.get_error() < 3 and model.get_error() > 1

def test_get_error_many_points_and_dims():
    """Model with high n and high dims should have lots of error."""
    model = kmeans.KMeans(k=2, dims=100)
    mat, _ = utils.make_clusters(n=100, m=100, k=2)
    model.fit(mat)
    assert model.get_error() < 100 and model.get_error() > 50

def test_distance_method():
    """Confirm distance method works as intended with Pythagorean Theorem."""
    x_1 = np.array([3,4])
    x_2 = np.array([0,0])
    model = kmeans.KMeans(k=2, dims=2)
    dist = model._get_distance(x_1, x_2)
    assert dist == 5

# Fit/Predict test cases - again we're going off vibes but printing figures
# out and putting them in the subdirectory to validate the assertions are
# good.

def test_fit_predict_high_scale_vs_low_scale():
    """
    Run the model with the same seed with different scaling factors. Confirm
    the error increases as scaling factor does. Save figures in the figures
    subdirectory (commented out to prevent testing issues).
    """
    error = []
    scale = [0.01, 0.1, 1, 3]
    for factor in scale:
        n = 1000
        m = 2
        k = 3
        model = kmeans.KMeans(k=3, dims=2)
        mat, truth = utils.make_clusters(n=n, m=m, k=k, scale=factor, seed=161803)
        model.fit(mat)
        pred = model.predict(mat)
        sil = silhouette.Silhouette()
        """
        utils.plot_multipanel(
            mat=mat,
            truth=truth,
            pred=pred,
            score=sil.score(mat, pred),
            filename=f"figures/scale_test_{str(factor)}.png"
        )
        """
        error.append(model.get_error())
    assert error[0] < error[1] and error[1] < error[2] and error[2] < error[3]

def test_predict_before_model():
    """Model should not precict if it hasn't been fit yet."""
    model = kmeans.KMeans(k=3, dims=2)
    mat, _ = utils.make_clusters()
    with pytest.raises(ValueError):
        _ = model.predict(mat)