import numpy as np

def test_covariance_matrix():
    returns = np.random.rand(100, 4)
    cov_matrix = np.cov(returns.T)
    assert cov_matrix.shape == (4, 4)