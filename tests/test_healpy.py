import unittest.mock

import numpy as np
import numpy.testing as npt
import pytest

try:
    import healpy as hp
except ImportError:
    HAVE_HEALPY = False
else:
    HAVE_HEALPY = True

skipif_not_healpy = pytest.mark.skipif(not HAVE_HEALPY, reason="test requires healpy")


@skipif_not_healpy
def test_healpix_maps(rng):
    from heracles.healpy import HealpixMapper
    from heracles.mapper import Mapper

    nside = 1 << rng.integers(1, 10)
    npix = hp.nside2npix(nside)
    lmax = 1 << rng.integers(1, 10)
    deconv = rng.choice([True, False])

    mapper = HealpixMapper(nside, lmax, deconvolve=deconv)

    assert isinstance(mapper, Mapper)

    assert mapper.nside == nside
    assert mapper.lmax == lmax
    assert mapper.deconvolve == deconv
    assert mapper.area == hp.nside2pixarea(nside)

    # create a map
    m = mapper.create(1, 2, 3, spin=-3)

    assert m.shape == (1, 2, 3, npix)
    assert m.dtype.metadata == {
        "geometry": "healpix",
        "kernel": "healpix",
        "nside": nside,
        "lmax": lmax,
        "deconv": deconv,
        "spin": -3,
    }

    # random values for mapping
    size = 1000
    lon = rng.uniform(0, 360, size=size)
    lat = np.degrees(np.arcsin(rng.uniform(-1, 1, size=size)))
    x = rng.standard_normal(size=size)
    y = rng.standard_normal(size=size)

    # pixel indices of random positions
    ipix = hp.ang2pix(nside, lon, lat, lonlat=True)

    # map one set of values

    m = mapper.create()
    mapper.map_values(lon, lat, m, x)

    expected = np.zeros(npix)
    np.add.at(expected, ipix, x)

    npt.assert_array_equal(m, expected)

    # map two sets of values

    m = mapper.create(2)
    mapper.map_values(lon, lat, m, np.stack([x, y]))

    expected = np.zeros((2, npix))
    np.add.at(expected[0], ipix, x)
    np.add.at(expected[1], ipix, y)

    npt.assert_array_equal(m, expected)


@skipif_not_healpy
@unittest.mock.patch("healpy.map2alm")
def test_healpix_transform(mock_map2alm, rng):
    from heracles.core import update_metadata
    from heracles.healpy import HealpixMapper

    nside = 32
    npix = 12 * nside**2

    mapper = HealpixMapper(nside)

    # single scalar map
    m = rng.standard_normal(npix)
    update_metadata(m, spin=0, nside=nside, a=1)

    mock_map2alm.return_value = np.empty(0, dtype=complex)

    alms = mapper.transform(m)

    assert alms is mock_map2alm.return_value
    assert alms.dtype.metadata["spin"] == 0
    assert alms.dtype.metadata["a"] == 1
    assert alms.dtype.metadata["nside"] == nside

    # polarisation map
    m = rng.standard_normal((2, npix))
    update_metadata(m, spin=2, nside=nside, b=2)

    mock_map2alm.return_value = np.empty((3, 0), dtype=complex)

    alms = mapper.transform(m)

    assert alms.shape == (2, 0)
    assert alms.dtype.metadata["spin"] == 2
    assert alms.dtype.metadata["b"] == 2
    assert alms.dtype.metadata["nside"] == nside


@skipif_not_healpy
@unittest.mock.patch("healpy.map2alm")
def test_healpix_deconvolve(mock_map2alm):
    from heracles.core import update_metadata
    from heracles.healpy import HealpixMapper

    nside = 32
    npix = 12 * nside**2

    lmax = 48
    nlm = (lmax + 1) * (lmax + 2) // 2

    pw0, pw2 = hp.pixwin(nside, lmax=lmax, pol=True)
    pw2[:2] = 1.0

    mapper = HealpixMapper(nside, lmax, deconvolve=True)

    # single scalar map
    data = np.zeros(npix)
    update_metadata(data, spin=0)

    mock_map2alm.return_value = np.ones(nlm, dtype=complex)

    alm = mapper.transform(data)

    assert alm.shape == (nlm,)
    stop = 0
    for m in range(lmax + 1):
        start, stop = stop, stop + lmax - m + 1
        npt.assert_array_equal(alm[start:stop], 1.0 / pw0[m:])

    # polarisation map
    data = np.zeros((2, npix))
    update_metadata(data, spin=2)

    mock_map2alm.return_value = np.ones((3, nlm), dtype=complex)

    alm = mapper.transform(data)

    assert alm.shape == (2, nlm)
    stop = 0
    for m in range(lmax + 1):
        start, stop = stop, stop + lmax - m + 1
        npt.assert_array_equal(alm[0, start:stop], 1.0 / pw2[m:])
        npt.assert_array_equal(alm[1, start:stop], 1.0 / pw2[m:])
