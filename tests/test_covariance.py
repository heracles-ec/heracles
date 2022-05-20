import numpy as np
import pytest


def test_sample_covariance():
    from le3_pk_wl.covariance import SampleCovariance, add_sample

    n = 10
    size = 3
    size2 = 5

    samples = [np.random.randn(size) for _ in range(n)]
    samples2 = [np.random.randn(size2) for _ in range(n)]

    cov = SampleCovariance(size)

    assert cov.shape == (size, size)

    for x in samples:
        add_sample(cov, x)

    assert cov.sample_count == len(samples)
    np.testing.assert_allclose(cov.sample_row_mean, np.mean(samples, axis=0))
    np.testing.assert_allclose(cov.sample_col_mean, np.mean(samples, axis=0))

    x = np.reshape(samples, (n, size, 1))
    y = np.reshape(samples, (n, 1, size))
    cxy = n/(n-1)*((x - x.mean(axis=0))*(y - y.mean(axis=0))).mean(axis=0)

    assert np.allclose(cov, cxy)

    with pytest.raises(ValueError):
        add_sample(cov, np.zeros(size + 1))

    cov = SampleCovariance(size, size2)

    assert cov.shape == (size, size2)

    for x, y in zip(samples, samples2):
        add_sample(cov, x, y)

    assert cov.sample_count == len(samples)
    np.testing.assert_allclose(cov.sample_row_mean, np.mean(samples, axis=0))
    np.testing.assert_allclose(cov.sample_col_mean, np.mean(samples2, axis=0))

    x = np.reshape(samples, (n, size, 1))
    y = np.reshape(samples2, (n, 1, size2))
    cxy = n/(n-1)*((x - x.mean(axis=0))*(y - y.mean(axis=0))).mean(axis=0)

    assert np.allclose(cov, cxy)

    with pytest.raises(ValueError):
        add_sample(cov, np.zeros(size + 1), np.zeros(size2 - 1))


def test_update_covariance():
    from itertools import combinations_with_replacement
    from le3_pk_wl.covariance import update_covariance

    n = 4

    cov = {}

    sample = {i: np.random.randn(i+1) for i in range(n)}
    update_covariance(cov, sample)

    assert len(cov) == n*(n+1)//2
    for k1, k2 in combinations_with_replacement(sample, 2):
        assert (k1, k2) in cov
        assert cov[k1, k2].shape == (sample[k1].size, sample[k2].size)
        assert np.all(cov[k1, k2] == 0)

    sample2 = {i: np.random.randn(i+1) for i in range(n)}
    update_covariance(cov, sample2)

    assert len(cov) == n*(n+1)//2
    for k1, k2 in combinations_with_replacement(sample, 2):
        assert (k1, k2) in cov
        assert cov[k1, k2].shape == (sample[k1].size, sample[k2].size)
        assert np.all(cov[k1, k2] != 0)
