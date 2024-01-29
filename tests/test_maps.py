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


@unittest.mock.patch.dict("heracles.maps._mapper._KERNELS", clear=True)
def test_mapper_from_dict():
    from heracles.maps import mapper_from_dict
    from heracles.maps._mapper import _KERNELS

    mock = unittest.mock.Mock()

    assert _KERNELS == {}
    _KERNELS["test"] = mock

    d = {"kernel": "test", "a": 1}

    mapper_from_dict(d)
    mock.from_dict.assert_called_once_with(d)


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
    mapper.map_values(lon, lat, [m])

    expected = np.zeros(npix)
    np.add.at(expected, ipix, 1)

    npt.assert_array_equal(m, expected)

    # map positions with weights

    m = np.zeros(mapper.size, mapper.dtype)
    mapper.map_values(lon, lat, [m], None, w)

    expected = np.zeros(npix)
    np.add.at(expected, ipix, w)

    npt.assert_array_equal(m, expected)

    # map one set of values

    m = np.zeros(mapper.size, mapper.dtype)
    mapper.map_values(lon, lat, [m], [x])

    expected = np.zeros(npix)
    np.add.at(expected, ipix, x)

    npt.assert_array_equal(m, expected)

    # map two sets of values

    m = np.zeros((2, mapper.size), mapper.dtype)
    mapper.map_values(lon, lat, [m[0], m[1]], [x, y])

    expected = np.zeros((2, npix))
    np.add.at(expected[0], ipix, x)
    np.add.at(expected[1], ipix, y)

    npt.assert_array_equal(m, expected)

    # map one set of values with weights

    m = np.zeros(mapper.size, mapper.dtype)
    mapper.map_values(lon, lat, [m], [x], w)

    expected = np.zeros(npix)
    np.add.at(expected, ipix, w * x)

    npt.assert_array_equal(m, expected)

    # map two sets of values with weights

    m = np.zeros((2, mapper.size), mapper.dtype)
    mapper.map_values(lon, lat, [m[0], m[1]], [x, y], w)

    expected = np.zeros((2, npix))
    np.add.at(expected[0], ipix, w * x)
    np.add.at(expected[1], ipix, w * y)

    npt.assert_array_equal(m, expected)

    # test from_dict
    mapper = Healpix.from_dict({"nside": 12})
    assert isinstance(mapper, Healpix)
    assert mapper.nside == 12


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


@unittest.mock.patch.dict("heracles.maps._mapper._KERNELS", clear=True)
@unittest.mock.patch("heracles.maps._mapping.deconvolved")
def test_transform_maps(mock_deconvolved, rng):
    import numpy as np

    from heracles.maps import transform_maps
    from heracles.maps._mapper import _KERNELS

    alms_x = unittest.mock.Mock()
    alms_ye = unittest.mock.Mock()
    alms_yb = unittest.mock.Mock()

    mock = unittest.mock.Mock()
    mock.from_dict().transform.side_effect = (alms_x, (alms_ye, alms_yb))
    _KERNELS["test"] = mock

    dtype = np.dtype(float, metadata={"kernel": "test"})

    maps = {
        ("X", 0): unittest.mock.Mock(dtype=dtype),
        ("Y", 1): unittest.mock.Mock(dtype=dtype),
    }

    alms = transform_maps(maps)

    assert len(alms) == 3
    assert alms.keys() == {("X", 0), ("Y_E", 1), ("Y_B", 1)}
    assert alms["X", 0] is alms_x
    assert alms["Y_E", 1] is alms_ye
    assert alms["Y_B", 1] is alms_yb
