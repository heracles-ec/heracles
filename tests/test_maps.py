import unittest.mock

import numpy as np
import numpy.testing as npt
import pytest


@unittest.mock.patch.dict("heracles.maps._mapper._KERNELS", clear=True)
def test_kernel_registry():
    from heracles.maps import Mapper, get_kernels

    class TestMapper(Mapper, kernel="test"):
        pass

    assert get_kernels() == {"test": TestMapper}
    assert Mapper.kernel is None
    assert TestMapper.kernel == "test"


def test_healpix_maps(rng):
    import healpy as hp

    from heracles.maps import Healpix

    nside = 1 << rng.integers(1, 10)
    npix = hp.nside2npix(nside)

    mapper = Healpix(nside)

    assert mapper.metadata == {"kernel": "healpix", "nside": nside}
    assert mapper.nside == nside
    assert mapper.area == hp.nside2pixarea(nside)
    assert mapper.dtype == np.float64
    assert mapper.size == npix

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

    m = np.zeros(mapper.size, mapper.dtype)
    mapper(lon, lat, [m])

    expected = np.zeros(npix)
    np.add.at(expected, ipix, 1)

    npt.assert_array_equal(m, expected)

    # map positions with weights

    m = np.zeros(mapper.size, mapper.dtype)
    mapper(lon, lat, [m], None, w)

    expected = np.zeros(npix)
    np.add.at(expected, ipix, w)

    npt.assert_array_equal(m, expected)

    # map one set of values

    m = np.zeros(mapper.size, mapper.dtype)
    mapper(lon, lat, [m], [x])

    expected = np.zeros(npix)
    np.add.at(expected, ipix, x)

    npt.assert_array_equal(m, expected)

    # map two sets of values

    m = np.zeros((2, mapper.size), mapper.dtype)
    mapper(lon, lat, [m[0], m[1]], [x, y])

    expected = np.zeros((2, npix))
    np.add.at(expected[0], ipix, x)
    np.add.at(expected[1], ipix, y)

    npt.assert_array_equal(m, expected)

    # map one set of values with weights

    m = np.zeros(mapper.size, mapper.dtype)
    mapper(lon, lat, [m], [x], w)

    expected = np.zeros(npix)
    np.add.at(expected, ipix, w * x)

    npt.assert_array_equal(m, expected)

    # map two sets of values with weights

    m = np.zeros((2, mapper.size), mapper.dtype)
    mapper(lon, lat, [m[0], m[1]], [x, y], w)

    expected = np.zeros((2, npix))
    np.add.at(expected[0], ipix, w * x)
    np.add.at(expected[1], ipix, w * y)

    npt.assert_array_equal(m, expected)


class MockField:
    def __init__(self):
        self.args = []
        self.return_value = object()

    async def __call__(self, catalog, mapper, *, progress=None):
        self.args.append((catalog, mapper))
        return self.return_value

    def assert_called_with(self, *args):
        assert self.args[-1] == args

    def assert_any_call(self, *args):
        assert args in self.args


class MockCatalog:
    size = 10
    page_size = 1

    def __iter__(self):
        for i in range(0, self.size, self.page_size):
            yield {}


@pytest.mark.parametrize("parallel", [False, True])
def test_map_catalogs(parallel):
    from heracles.maps import map_catalogs

    mapper = unittest.mock.Mock()

    fields = {"a": MockField(), "b": MockField(), "z": MockField()}
    catalogs = {"x": MockCatalog(), "y": MockCatalog()}

    maps = map_catalogs(mapper, fields, catalogs, parallel=parallel)

    for k in fields:
        for i in catalogs:
            fields[k].assert_any_call(catalogs[i], mapper)
            assert maps[k, i] is fields[k].return_value


def test_map_catalogs_match():
    from heracles.maps import map_catalogs

    mapper = unittest.mock.Mock()
    fields = {"a": MockField(), "b": MockField(), "c": MockField()}
    catalogs = {"x": MockCatalog(), "y": MockCatalog()}

    maps = map_catalogs(mapper, fields, catalogs, include=[(..., "y")])

    assert set(maps.keys()) == {("a", "y"), ("b", "y"), ("c", "y")}

    maps = map_catalogs(mapper, fields, catalogs, exclude=[("a", ...)])

    assert set(maps.keys()) == {("b", "x"), ("b", "y"), ("c", "x"), ("c", "y")}


def test_transform_maps(rng):
    from heracles.core import update_metadata
    from heracles.maps import transform_maps

    nside = 32
    npix = 12 * nside**2

    t = rng.standard_normal(npix)
    update_metadata(t, spin=0, nside=nside, a=1)
    p = rng.standard_normal((2, npix))
    update_metadata(p, spin=2, nside=nside, b=2)

    # single scalar map
    maps = {("T", 0): t}
    alms = transform_maps(maps)

    assert len(alms) == 1
    assert alms.keys() == maps.keys()
    assert alms["T", 0].dtype.metadata["spin"] == 0
    assert alms["T", 0].dtype.metadata["a"] == 1
    assert alms["T", 0].dtype.metadata["nside"] == nside

    # polarisation map
    maps = {("P", 0): p}
    alms = transform_maps(maps)

    assert len(alms) == 2
    assert alms.keys() == {("P_E", 0), ("P_B", 0)}
    assert alms["P_E", 0].dtype.metadata["spin"] == 2
    assert alms["P_B", 0].dtype.metadata["spin"] == 2
    assert alms["P_E", 0].dtype.metadata["b"] == 2
    assert alms["P_B", 0].dtype.metadata["b"] == 2
    assert alms["P_E", 0].dtype.metadata["nside"] == nside
    assert alms["P_B", 0].dtype.metadata["nside"] == nside

    # mixed
    maps = {("T", 0): t, ("P", 1): p}
    alms = transform_maps(maps)

    assert len(alms) == 3
    assert alms.keys() == {("T", 0), ("P_E", 1), ("P_B", 1)}
    assert alms["T", 0].dtype.metadata["spin"] == 0
    assert alms["P_E", 1].dtype.metadata["spin"] == 2
    assert alms["P_B", 1].dtype.metadata["spin"] == 2
    assert alms["T", 0].dtype.metadata["a"] == 1
    assert alms["P_E", 1].dtype.metadata["b"] == 2
    assert alms["P_B", 1].dtype.metadata["b"] == 2
    assert alms["T", 0].dtype.metadata["nside"] == nside
    assert alms["P_E", 1].dtype.metadata["nside"] == nside
    assert alms["P_B", 1].dtype.metadata["nside"] == nside

    # explicit lmax per map
    maps = {("T", 0): t, ("P", 1): p}
    lmax = {"T": 10, "P": 20}
    alms = transform_maps(maps, lmax=lmax)

    assert len(alms) == 3
    assert alms.keys() == {("T", 0), ("P_E", 1), ("P_B", 1)}
    assert alms["T", 0].size == (lmax["T"] + 1) * (lmax["T"] + 2) // 2
    assert alms["P_E", 1].size == (lmax["P"] + 1) * (lmax["P"] + 2) // 2
    assert alms["P_B", 1].size == (lmax["P"] + 1) * (lmax["P"] + 2) // 2
