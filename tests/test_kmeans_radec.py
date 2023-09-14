import pytest


@pytest.mark.flaky(reruns=2)
def test_kmeans_sample(random_generator):
    import numpy as np

    from heracles._kmeans_radec import kmeans_sample

    npts = 10000
    ncen = 20

    pts = np.empty((npts, 2))
    pts[:, 0] = random_generator.uniform(-180, 180, size=npts)
    pts[:, 1] = np.degrees(np.arcsin(random_generator.uniform(-1, 1, size=npts)))

    km = kmeans_sample(pts, ncen, verbose=2)

    assert km.converged
    assert len(km.centers) == ncen
    assert len(km.labels) == npts
    assert len(km.distances) == npts
    assert len(km.r50) == ncen * 2
    assert len(km.r90) == ncen * 2
