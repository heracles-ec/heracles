import numpy as np
import pytest


def test_sample_covariance(rng):
    from heracles.covariance import SampleCovariance, add_sample

    n = 10
    size = 3
    size2 = 5

    samples = [rng.standard_normal(size) for _ in range(n)]
    samples2 = [rng.standard_normal(size2) for _ in range(n)]

    cov = SampleCovariance(size)

    assert cov.shape == (size, size)

    for x in samples:
        add_sample(cov, x)

    assert cov.sample_count == len(samples)
    np.testing.assert_allclose(cov.sample_row_mean, np.mean(samples, axis=0))
    np.testing.assert_allclose(cov.sample_col_mean, np.mean(samples, axis=0))

    x = np.reshape(samples, (n, size, 1))
    y = np.reshape(samples, (n, 1, size))
    cxy = n / (n - 1) * ((x - x.mean(axis=0)) * (y - y.mean(axis=0))).mean(axis=0)

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
    cxy = n / (n - 1) * ((x - x.mean(axis=0)) * (y - y.mean(axis=0))).mean(axis=0)

    assert np.allclose(cov, cxy)

    with pytest.raises(ValueError):
        add_sample(cov, np.zeros(size + 1), np.zeros(size2 - 1))


def test_update_covariance(rng):
    from itertools import combinations_with_replacement

    from heracles.covariance import update_covariance

    n = 4

    cov = {}

    sample = {i: rng.standard_normal(i + 1) for i in range(n)}
    update_covariance(cov, sample)

    assert len(cov) == n * (n + 1) // 2
    for k1, k2 in combinations_with_replacement(sample, 2):
        assert (k1, k2) in cov
        assert cov[k1, k2].shape == (sample[k1].size, sample[k2].size)
        assert np.all(cov[k1, k2] == 0)

    sample2 = {i: rng.standard_normal(i + 1) for i in range(n)}
    update_covariance(cov, sample2)

    assert len(cov) == n * (n + 1) // 2
    for k1, k2 in combinations_with_replacement(sample, 2):
        assert (k1, k2) in cov
        assert cov[k1, k2].shape == (sample[k1].size, sample[k2].size)
        assert np.all(cov[k1, k2] != 0)


def test_jackknife_regions_kmeans():
    from heracles.covariance import jackknife_regions_kmeans

    nside = 64

    fpmap = np.zeros(12 * nside**2)
    fpmap[: fpmap.size // 2] = 1.0

    n = 12

    jkmap, jkcen = jackknife_regions_kmeans(fpmap, n, return_centers=True)

    assert jkmap.size == fpmap.size
    assert np.all(jkmap[jkmap.size // 2 :] == 0)
    assert np.all(jkmap[: jkmap.size // 2] > 0)
    assert list(np.unique(jkmap)) == list(range(n + 1))
    assert len(jkcen) == n
