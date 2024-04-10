import unittest.mock

import numpy as np
import numpy.testing as npt
import pytest


def test_healpix_maps(rng):
    import healpy as hp

    from heracles.maps import Healpix, Mapper

    nside = 1 << rng.integers(1, 10)
    npix = hp.nside2npix(nside)
    lmax = 1 << rng.integers(1, 10)
    deconv = rng.choice([True, False])

    mapper = Healpix(nside, lmax, deconvolve=deconv)

    assert isinstance(mapper, Mapper)

    assert mapper.nside == nside
    assert mapper.lmax == lmax
    assert mapper.deconvolve == deconv
    assert mapper.area == hp.nside2pixarea(nside)

    # create a map
    m = mapper.create(1, 2, 3, dtype=np.uint16, spin=-3)

    assert m.shape == (1, 2, 3, npix)
    assert m.dtype == np.uint16
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
    w = 10 ** rng.standard_normal(size=size)

    # pixel indices of random positions
    ipix = hp.ang2pix(nside, lon, lat, lonlat=True)

    # map positions

    m = mapper.create()
    mapper.map_values(lon, lat, [m])

    expected = np.zeros(npix)
    np.add.at(expected, ipix, 1)

    npt.assert_array_equal(m, expected)

    # map positions with weights

    m = mapper.create()
    mapper.map_values(lon, lat, [m], None, w)

    expected = np.zeros(npix)
    np.add.at(expected, ipix, w)

    npt.assert_array_equal(m, expected)

    # map one set of values

    m = mapper.create()
    mapper.map_values(lon, lat, [m], [x])

    expected = np.zeros(npix)
    np.add.at(expected, ipix, x)

    npt.assert_array_equal(m, expected)

    # map two sets of values

    m = mapper.create(2)
    mapper.map_values(lon, lat, [m[0], m[1]], [x, y])

    expected = np.zeros((2, npix))
    np.add.at(expected[0], ipix, x)
    np.add.at(expected[1], ipix, y)

    npt.assert_array_equal(m, expected)

    # map one set of values with weights

    m = mapper.create()
    mapper.map_values(lon, lat, [m], [x], w)

    expected = np.zeros(npix)
    np.add.at(expected, ipix, w * x)

    npt.assert_array_equal(m, expected)

    # map two sets of values with weights

    m = mapper.create(2)
    mapper.map_values(lon, lat, [m[0], m[1]], [x, y], w)

    expected = np.zeros((2, npix))
    np.add.at(expected[0], ipix, w * x)
    np.add.at(expected[1], ipix, w * y)

    npt.assert_array_equal(m, expected)


@unittest.mock.patch("healpy.map2alm")
def test_healpix_transform(mock_map2alm, rng):
    from heracles.core import update_metadata
    from heracles.maps import Healpix

    nside = 32
    npix = 12 * nside**2

    mapper = Healpix(nside)

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

    mock_map2alm.return_value = (
        np.empty(0, dtype=complex),
        np.empty(0, dtype=complex),
        np.empty(0, dtype=complex),
    )

    alms = mapper.transform(m)

    assert len(alms) == 2
    assert alms[0] is mock_map2alm.return_value[1]
    assert alms[1] is mock_map2alm.return_value[2]
    assert alms[0].dtype.metadata["spin"] == 2
    assert alms[1].dtype.metadata["spin"] == 2
    assert alms[0].dtype.metadata["b"] == 2
    assert alms[1].dtype.metadata["b"] == 2
    assert alms[0].dtype.metadata["nside"] == nside
    assert alms[1].dtype.metadata["nside"] == nside


class MockCatalog:
    size = 10
    page_size = 1

    def __iter__(self):
        for i in range(0, self.size, self.page_size):
            yield {}


@pytest.mark.parametrize("parallel", [False, True])
def test_map_catalogs(parallel):
    from unittest.mock import AsyncMock

    from heracles.maps import map_catalogs

    fields = {"a": AsyncMock(), "b": AsyncMock(), "z": AsyncMock()}
    catalogs = {"x": MockCatalog(), "y": MockCatalog()}

    maps = map_catalogs(fields, catalogs, parallel=parallel)

    for k in fields:
        for i in catalogs:
            fields[k].assert_any_call(catalogs[i], progress=None)
            assert maps[k, i] is fields[k].return_value


def test_map_catalogs_match():
    from unittest.mock import AsyncMock

    from heracles.maps import map_catalogs

    fields = {"a": AsyncMock(), "b": AsyncMock(), "c": AsyncMock()}
    catalogs = {"x": MockCatalog(), "y": MockCatalog()}

    maps = map_catalogs(fields, catalogs, include=[(..., "y")])

    assert set(maps.keys()) == {("a", "y"), ("b", "y"), ("c", "y")}

    maps = map_catalogs(fields, catalogs, exclude=[("a", ...)])

    assert set(maps.keys()) == {("b", "x"), ("b", "y"), ("c", "x"), ("c", "y")}


def test_transform_maps(rng):
    from unittest.mock import Mock

    from heracles.maps import transform_maps

    x = Mock()
    y = Mock()
    x.mapper_or_error.transform.return_value = Mock()
    y.mapper_or_error.transform.return_value = (Mock(), Mock())

    fields = {"X": x, "Y": y}
    maps = {("X", 0): Mock(), ("Y", 1): Mock()}

    alms = transform_maps(fields, maps)

    assert len(alms) == 3
    assert alms.keys() == {("X", 0), ("Y_E", 1), ("Y_B", 1)}
    assert alms["X", 0] is x.mapper_or_error.transform.return_value
    assert alms["Y_E", 1] is y.mapper_or_error.transform.return_value[0]
    assert alms["Y_B", 1] is y.mapper_or_error.transform.return_value[1]
