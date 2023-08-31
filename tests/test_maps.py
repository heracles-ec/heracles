import numpy as np
import healpy as hp
import pytest

from .conftest import warns


def map_catalog(m, catalog):
    g = m(catalog)
    fn = next(g)
    for page in catalog:
        fn(page)
    try:
        next(g)
    except StopIteration as stop:
        return stop.value
    else:
        raise RuntimeError("generator did not stop")


@pytest.fixture
def nside():
    return 64


@pytest.fixture
def sigma_e():
    return 0.1


@pytest.fixture
def vmap(nside):
    return np.round(np.random.rand(12 * nside**2))


@pytest.fixture
def page(nside):
    from unittest.mock import Mock

    ipix = np.ravel(
        4 * hp.ring2nest(nside, np.arange(12 * nside**2))[:, np.newaxis]
        + [0, 1, 2, 3]
    )

    ra, dec = hp.pix2ang(nside * 2, ipix, nest=True, lonlat=True)

    size = ra.size

    w = np.random.rand(size // 4, 4)
    g1 = np.random.randn(size // 4, 4)
    g2 = np.random.randn(size // 4, 4)
    g1 -= np.sum(w * g1, axis=-1, keepdims=True) / np.sum(w, axis=-1, keepdims=True)
    g2 -= np.sum(w * g2, axis=-1, keepdims=True) / np.sum(w, axis=-1, keepdims=True)
    w, g1, g2 = w.reshape(-1), g1.reshape(-1), g2.reshape(-1)

    cols = {"ra": ra, "dec": dec, "g1": g1, "g2": g2, "w": w}

    def get(*names):
        return cols[names[0]] if len(names) == 1 else [cols[name] for name in names]

    page = Mock()
    page.size = size
    page.get = get
    page.__getitem__ = lambda self, *names: get(*names)

    return page


@pytest.fixture
def catalog(page):
    from unittest.mock import Mock

    catalog = Mock()
    catalog.visibility = None
    catalog.__iter__ = lambda self: iter([page])

    return catalog


def test_visibility_map(nside, vmap):
    from unittest.mock import Mock
    from heracles.maps import VisibilityMap

    fsky = vmap.mean()

    for nside_out in [nside // 2, nside, nside * 2]:
        catalog = Mock()
        catalog.visibility = vmap

        mapper = VisibilityMap(nside_out)

        with warns(UserWarning if nside != nside_out else None):
            result = mapper(catalog)

        assert result is not vmap

        assert result.shape == (12 * nside_out**2,)
        assert result.dtype.metadata == {"spin": 0, "kernel": "healpix", "power": 0}
        assert np.isclose(result.mean(), fsky)

    # test missing visibility map
    catalog = Mock()
    catalog.visibility = None
    mapper = VisibilityMap(nside)
    with pytest.raises(ValueError, match="no visibility"):
        mapper(catalog)


def test_position_map(nside, catalog, vmap):
    from heracles.maps import PositionMap

    # normal mode: compute overdensity maps with metadata

    m = map_catalog(PositionMap(nside, "ra", "dec"), catalog)

    assert m.shape == (12 * nside**2,)
    assert m.dtype.metadata == {"spin": 0, "nbar": 4.0, "kernel": "healpix", "power": 0}
    np.testing.assert_array_equal(m, 0)

    # compute number count map

    m = map_catalog(PositionMap(nside, "ra", "dec", overdensity=False), catalog)

    assert m.shape == (12 * nside**2,)
    assert m.dtype.metadata == {"spin": 0, "nbar": 4.0, "kernel": "healpix", "power": 1}
    np.testing.assert_array_equal(m, 4)

    # compute overdensity maps with visibility map

    catalog.visibility = vmap

    m = map_catalog(PositionMap(nside, "ra", "dec"), catalog)

    assert m.shape == (12 * nside**2,)
    assert m.dtype.metadata == {
        "spin": 0,
        "nbar": 4.0 / vmap.mean(),
        "kernel": "healpix",
        "power": 0,
    }

    # compute number count map with visibility map

    m = map_catalog(PositionMap(nside, "ra", "dec", overdensity=False), catalog)

    assert m.shape == (12 * nside**2,)
    assert m.dtype.metadata == {
        "spin": 0,
        "nbar": 4.0 / vmap.mean(),
        "kernel": "healpix",
        "power": 1,
    }


def test_scalar_map(nside, catalog):
    from heracles.maps import ScalarMap

    m = map_catalog(ScalarMap(nside, "ra", "dec", "g1", "w"), catalog)

    w = next(iter(catalog))["w"]
    w = w.reshape(w.size // 4, 4).sum(axis=-1)
    wbar = w.mean()

    assert m.shape == (12 * nside**2,)
    assert m.dtype.metadata == {
        "spin": 0,
        "wbar": wbar,
        "kernel": "healpix",
        "power": 0,
    }
    np.testing.assert_array_almost_equal(m, 0)

    m = map_catalog(ScalarMap(nside, "ra", "dec", "g1", "w", normalize=False), catalog)

    assert m.shape == (12 * nside**2,)
    assert m.dtype.metadata == {
        "spin": 0,
        "wbar": wbar,
        "kernel": "healpix",
        "power": 1,
    }
    np.testing.assert_array_almost_equal(m, 0)


def test_complex_map(nside, catalog):
    from heracles.maps import ComplexMap

    m = map_catalog(ComplexMap(nside, "ra", "dec", "g1", "g2", "w", spin=2), catalog)

    w = next(iter(catalog))["w"]
    w = w.reshape(w.size // 4, 4).sum(axis=-1)
    wbar = w.mean()

    assert m.shape == (
        2,
        12 * nside**2,
    )
    assert m.dtype.metadata == {
        "spin": 2,
        "wbar": wbar,
        "kernel": "healpix",
        "power": 0,
    }
    np.testing.assert_array_almost_equal(m, 0)

    m = map_catalog(
        ComplexMap(nside, "ra", "dec", "g1", "g2", "w", spin=1, normalize=False),
        catalog,
    )

    assert m.shape == (
        2,
        12 * nside**2,
    )
    assert m.dtype.metadata == {
        "spin": 1,
        "wbar": wbar,
        "kernel": "healpix",
        "power": 1,
    }
    np.testing.assert_array_almost_equal(m, 0)


def test_weight_map(nside, catalog):
    from heracles.maps import WeightMap

    m = map_catalog(WeightMap(nside, "ra", "dec", "w"), catalog)

    w = next(iter(catalog))["w"]
    w = w.reshape(w.size // 4, 4).sum(axis=-1)
    wbar = w.mean()

    assert m.shape == (12 * nside**2,)
    assert m.dtype.metadata == {
        "spin": 0,
        "wbar": wbar,
        "kernel": "healpix",
        "power": 0,
    }
    np.testing.assert_array_almost_equal(m, w / wbar)

    m = map_catalog(WeightMap(nside, "ra", "dec", "w", normalize=False), catalog)

    assert m.shape == (12 * nside**2,)
    assert m.dtype.metadata == {
        "spin": 0,
        "wbar": wbar,
        "kernel": "healpix",
        "power": 1,
    }
    np.testing.assert_array_almost_equal(m, w)


def test_transform_maps():
    from heracles.maps import transform_maps, update_metadata

    nside = 32
    npix = 12 * nside**2

    t = np.random.randn(npix)
    update_metadata(t, spin=0, a=1)
    p = np.random.randn(2, npix)
    update_metadata(p, spin=2, b=2)

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


def test_update_metadata():
    from heracles.maps import update_metadata

    a = np.empty(0)

    assert a.dtype.metadata is None

    update_metadata(a, x=1)

    assert a.dtype.metadata == {"x": 1}

    update_metadata(a, y=2)

    assert a.dtype.metadata == {"x": 1, "y": 2}

    update_metadata(a, x=3)

    assert a.dtype.metadata == {"x": 3, "y": 2}

    # check dtype fields are preserved

    a = np.array(
        [("Alice", 37, 56.0), ("Bob", 25, 73.0)],
        dtype=[("f0", "U10"), ("f1", "i4"), ("f2", "f4")],
    )

    a_fields_original = np.copy(a.dtype.fields)

    update_metadata(a, x=1)

    assert a.dtype.fields == a_fields_original
    assert a.dtype.metadata == {"x": 1}

    update_metadata(a, y=2)

    assert a.dtype.fields == a_fields_original
    assert a.dtype.metadata == {"x": 1, "y": 2}


class MockMap:
    def __init__(self):
        self.args = []
        self.return_value = object()

    def __call__(self, catalog):
        self.args.append(catalog)
        return self.return_value

    def assert_called_with(self, value):
        assert self.args[-1] is value

    def assert_any_call(self, value):
        assert value in self.args


class MockMapGen(MockMap):
    def __call__(self, catalog):
        def f(page):
            pass

        yield f
        return super().__call__(catalog)


class MockCatalog:
    size = 10
    page_size = 1

    def __iter__(self):
        for _ in range(0, self.size, self.page_size):
            yield {}


@pytest.mark.parametrize("Map", [MockMap, MockMapGen])
@pytest.mark.parametrize("parallel", [False, True])
def test_map_catalogs(Map, parallel):
    from heracles.maps import map_catalogs

    maps = {"a": Map(), "b": Map(), "z": Map()}
    catalogs = {"x": MockCatalog(), "y": MockCatalog()}

    data = map_catalogs(maps, catalogs, parallel=parallel)

    for k in maps:
        for i in catalogs:
            maps[k].assert_any_call(catalogs[i])
            assert data[k, i] is maps[k].return_value


@pytest.mark.parametrize("Map", [MockMap, MockMapGen])
def test_map_catalogs_match(Map):
    from heracles.maps import map_catalogs

    maps = {"a": Map(), "b": Map(), "c": Map()}
    catalogs = {"x": MockCatalog(), "y": MockCatalog()}

    data = map_catalogs(maps, catalogs, include=[(..., "y")])

    assert set(data.keys()) == {("a", "y"), ("b", "y"), ("c", "y")}

    data = map_catalogs(maps, catalogs, exclude=[("a", ...)])

    assert set(data.keys()) == {("b", "x"), ("b", "y"), ("c", "x"), ("c", "y")}
