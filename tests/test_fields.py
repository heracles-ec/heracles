import coroutines
import healpy as hp
import numpy as np
import pytest


@pytest.fixture
def nside():
    return 64


@pytest.fixture
def sigma_e():
    return 0.1


# use HEALPix mapper for testing
# TODO: mock mapper
@pytest.fixture
def mapper(nside):
    from heracles.maps import Healpix

    return Healpix(nside)


@pytest.fixture
def vmap(nside, rng):
    return np.round(rng.random(12 * nside**2))


@pytest.fixture
def page(nside, rng):
    from unittest.mock import Mock

    ipix = np.ravel(
        4 * hp.ring2nest(nside, np.arange(12 * nside**2))[:, np.newaxis]
        + [0, 1, 2, 3],
    )

    ra, dec = hp.pix2ang(nside * 2, ipix, nest=True, lonlat=True)

    size = ra.size

    w = rng.random((size // 4, 4))
    g1 = rng.standard_normal((size // 4, 4))
    g2 = rng.standard_normal((size // 4, 4))
    g1 -= np.sum(w * g1, axis=-1, keepdims=True) / np.sum(w, axis=-1, keepdims=True)
    g2 -= np.sum(w * g2, axis=-1, keepdims=True) / np.sum(w, axis=-1, keepdims=True)
    w, g1, g2 = w.reshape(-1), g1.reshape(-1), g2.reshape(-1)

    cols = {"ra": ra, "dec": dec, "g1": g1, "g2": g2, "w": w}

    def get(*names):
        if len(names) == 1:
            return cols[names[0]]
        return [cols[name] for name in names]

    page = Mock()
    page.size = size
    page.get = get
    page.__getitem__ = lambda self, *names: get(*names)

    return page


@pytest.fixture
def catalog(page):
    from unittest.mock import Mock

    catalog = Mock()
    catalog.size = page.size
    catalog.visibility = None
    catalog.metadata = {"catalog": catalog.label}
    catalog.__iter__ = lambda self: iter([page])

    return catalog


def test_field_abc():
    from unittest.mock import Mock

    from heracles.fields import Columns, Field

    with pytest.raises(TypeError):
        Field()

    class SpinLessField(Field):
        def _init_columns(self, *columns: str) -> Columns:
            return columns

        async def __call__(self):
            pass

    f = SpinLessField(None)

    with pytest.raises(ValueError, match="undefined spin weight"):
        f.spin

    class TestField(Field, spin=0):
        uses = "lon", "lat", "[weight]"

        async def __call__(self):
            pass

    f = TestField(None)

    assert f.mapper is None
    assert f.columns is None
    assert f.spin == 0

    with pytest.raises(ValueError):
        f.mapper_or_error

    with pytest.raises(ValueError):
        f.columns_or_error

    mapper = Mock()

    with pytest.raises(ValueError, match="accepts 2 to 3 columns"):
        TestField(mapper, "lon")

    f = TestField(mapper, "lon", "lat", mask="W")

    assert f.mapper is mapper
    assert f.columns == ("lon", "lat", None)
    assert f.mask == "W"


def test_visibility(nside, vmap):
    from contextlib import nullcontext
    from unittest.mock import Mock

    from heracles.fields import Visibility
    from heracles.maps import Healpix

    fsky = vmap.mean()

    for nside_out in [nside // 2, nside, nside * 2]:
        catalog = Mock()
        catalog.visibility = vmap
        catalog.metadata = {"catalog": catalog.label}

        mapper_out = Healpix(nside_out)

        f = Visibility(mapper_out)

        with pytest.warns(UserWarning) if nside != nside_out else nullcontext():
            result = coroutines.run(f(catalog))

        assert result is not vmap

        assert result.shape == (12 * nside_out**2,)
        assert result.dtype.metadata == {
            "catalog": catalog.label,
            "spin": 0,
            "geometry": "healpix",
            "kernel": "healpix",
            "nside": mapper_out.nside,
            "lmax": mapper_out.lmax,
            "deconv": mapper_out.deconvolve,
        }
        assert np.isclose(result.mean(), fsky)

    # test missing visibility map
    catalog = Mock()
    catalog.visibility = None
    f = Visibility(mapper)
    with pytest.raises(ValueError, match="no visibility"):
        coroutines.run(f(catalog))


def test_positions(mapper, catalog, vmap):
    from heracles.fields import Positions

    # bias
    npix = 12 * mapper.nside**2
    bias = (4 * np.pi / npix) * (catalog.size / npix)

    # normal mode: compute overdensity maps with metadata

    f = Positions(mapper, "ra", "dec")

    # test some default settings
    assert f.spin == 0
    assert f.overdensity
    assert f.nbar is None

    # create map
    m = coroutines.run(f(catalog))

    nbar = 4.0
    assert m.shape == (npix,)
    assert m.dtype.metadata == {
        "catalog": catalog.label,
        "spin": 0,
        "nbar": nbar,
        "geometry": "healpix",
        "kernel": "healpix",
        "nside": mapper.nside,
        "lmax": mapper.lmax,
        "deconv": mapper.deconvolve,
        "bias": pytest.approx(bias / nbar**2),
    }
    np.testing.assert_array_equal(m, 0)

    # compute number count map

    f = Positions(mapper, "ra", "dec", overdensity=False)
    m = coroutines.run(f(catalog))

    assert m.shape == (npix,)
    assert m.dtype.metadata == {
        "catalog": catalog.label,
        "spin": 0,
        "nbar": 4.0,
        "geometry": "healpix",
        "kernel": "healpix",
        "nside": mapper.nside,
        "lmax": mapper.lmax,
        "deconv": mapper.deconvolve,
        "bias": pytest.approx(bias / nbar**2),
    }
    np.testing.assert_array_equal(m, 1.0)

    # compute overdensity maps with visibility map

    catalog.visibility = vmap
    nbar /= vmap.mean()

    f = Positions(mapper, "ra", "dec")
    m = coroutines.run(f(catalog))

    assert m.shape == (12 * mapper.nside**2,)
    assert m.dtype.metadata == {
        "catalog": catalog.label,
        "spin": 0,
        "nbar": pytest.approx(nbar),
        "geometry": "healpix",
        "kernel": "healpix",
        "nside": mapper.nside,
        "lmax": mapper.lmax,
        "deconv": mapper.deconvolve,
        "bias": pytest.approx(bias / nbar**2),
    }

    # compute number count map with visibility map

    f = Positions(mapper, "ra", "dec", overdensity=False)
    m = coroutines.run(f(catalog))

    assert m.shape == (12 * mapper.nside**2,)
    assert m.dtype.metadata == {
        "catalog": catalog.label,
        "spin": 0,
        "nbar": pytest.approx(nbar),
        "geometry": "healpix",
        "kernel": "healpix",
        "nside": mapper.nside,
        "lmax": mapper.lmax,
        "deconv": mapper.deconvolve,
        "bias": pytest.approx(bias / nbar**2),
    }

    # compute overdensity maps with given (incorrect) nbar

    f = Positions(mapper, "ra", "dec", nbar=2 * nbar)
    with pytest.warns(UserWarning, match="mean density"):
        m = coroutines.run(f(catalog))

    assert m.dtype.metadata["nbar"] == 2 * nbar
    assert m.dtype.metadata["bias"] == pytest.approx(bias / (2 * nbar) ** 2)


def test_scalar_field(mapper, catalog):
    from heracles.fields import ScalarField

    npix = 12 * mapper.nside**2

    f = ScalarField(mapper, "ra", "dec", "g1", "w")
    m = coroutines.run(f(catalog))

    w = next(iter(catalog))["w"]
    v = next(iter(catalog))["g1"]
    v2 = ((w * v) ** 2).sum()
    w = w.reshape(w.size // 4, 4).sum(axis=-1)
    wbar = w.mean()
    bias = (4 * np.pi / npix / npix) * v2

    assert m.shape == (npix,)
    assert m.dtype.metadata == {
        "catalog": catalog.label,
        "spin": 0,
        "wbar": pytest.approx(wbar),
        "geometry": "healpix",
        "kernel": "healpix",
        "nside": mapper.nside,
        "lmax": mapper.lmax,
        "deconv": mapper.deconvolve,
        "bias": pytest.approx(bias / wbar**2),
    }
    np.testing.assert_array_almost_equal(m, 0)


def test_complex_field(mapper, catalog):
    from heracles.fields import Spin2Field

    npix = 12 * mapper.nside**2

    f = Spin2Field(mapper, "ra", "dec", "g1", "g2", "w")
    m = coroutines.run(f(catalog))

    w = next(iter(catalog))["w"]
    re = next(iter(catalog))["g1"]
    im = next(iter(catalog))["g2"]
    v2 = ((w * re) ** 2 + (w * im) ** 2).sum()
    w = w.reshape(w.size // 4, 4).sum(axis=-1)
    wbar = w.mean()
    bias = (4 * np.pi / npix / npix) * v2 / 2

    assert m.shape == (2, npix)
    assert m.dtype.metadata == {
        "catalog": catalog.label,
        "spin": 2,
        "wbar": pytest.approx(wbar),
        "geometry": "healpix",
        "kernel": "healpix",
        "nside": mapper.nside,
        "lmax": mapper.lmax,
        "deconv": mapper.deconvolve,
        "bias": pytest.approx(bias / wbar**2),
    }
    np.testing.assert_array_almost_equal(m, 0)


def test_weights(mapper, catalog):
    from heracles.fields import Weights

    npix = 12 * mapper.nside**2

    f = Weights(mapper, "ra", "dec", "w")
    m = coroutines.run(f(catalog))

    w = next(iter(catalog))["w"]
    v2 = (w**2).sum()
    w = w.reshape(w.size // 4, 4).sum(axis=-1)
    wbar = w.mean()
    bias = (4 * np.pi / npix / npix) * v2

    assert m.shape == (12 * mapper.nside**2,)
    assert m.dtype.metadata == {
        "catalog": catalog.label,
        "spin": 0,
        "wbar": pytest.approx(wbar),
        "geometry": "healpix",
        "kernel": "healpix",
        "nside": mapper.nside,
        "lmax": mapper.lmax,
        "deconv": mapper.deconvolve,
        "bias": pytest.approx(bias / wbar**2),
    }
    np.testing.assert_array_almost_equal(m, w / wbar)


def test_get_masks():
    from unittest.mock import Mock

    from heracles.fields import get_masks

    fields = {
        "A": Mock(mask="X", spin=0),
        "B": Mock(mask="Y", spin=2),
        "C": Mock(mask=None),
    }

    masks = get_masks(fields)

    assert masks == ["X", "Y"]

    masks = get_masks(fields, comb=1)

    assert masks == [("X",), ("Y",)]

    masks = get_masks(fields, comb=2)

    assert masks == [("X", "X"), ("X", "Y"), ("Y", "Y")]

    masks = get_masks(fields, comb=2, include=[("A",)])

    assert masks == [("X", "X"), ("X", "Y")]

    masks = get_masks(fields, comb=2, exclude=[("A", "B")])

    assert masks == [("X", "X"), ("Y", "Y")]

    masks = get_masks(fields, comb=2, include=[("A", "B")], append_eb=True)

    assert masks == []

    masks = get_masks(fields, comb=2, include=[("A", "B_E")], append_eb=True)

    assert masks == [("X", "Y")]
