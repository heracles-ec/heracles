import unittest.mock
import numpy as np
import healpy as hp
import pytest


@pytest.fixture
def nside():
    return 64


@pytest.fixture
def sigma_e():
    return 0.1


@pytest.fixture
def vmap(nside):
    return np.round(np.random.rand(12*nside**2))


@pytest.fixture
def catalog(nside):

    ipix = np.ravel(4*hp.ring2nest(nside, np.arange(12*nside**2))[:, np.newaxis] + [0, 1, 2, 3])

    ra, dec = hp.pix2ang(nside*2, ipix, nest=True, lonlat=True)

    w = np.random.rand(ra.size//4, 4)
    g1 = np.random.randn(ra.size//4, 4)
    g2 = np.random.randn(ra.size//4, 4)
    g1 -= np.sum(w*g1, axis=-1, keepdims=True)/np.sum(w, axis=-1, keepdims=True)
    g2 -= np.sum(w*g2, axis=-1, keepdims=True)/np.sum(w, axis=-1, keepdims=True)
    w, g1, g2 = w.reshape(-1), g1.reshape(-1), g2.reshape(-1)

    class MockCatalog:
        def __iter__(self):
            rows = unittest.mock.Mock()
            rows.size = ra.size
            rows.ra = ra
            rows.dec = dec
            rows.g1 = g1
            rows.g2 = g2
            rows.w = w
            yield rows

    return MockCatalog()


def test_visibility_map(nside):

    from le3_pk_wl.maps import visibility_map

    vmap = np.random.rand(12*nside**2)
    fsky = vmap.mean()

    for nside_out in [nside//2, nside, nside*2]:
        m = visibility_map(nside_out, vmap)

        assert m is not vmap

        assert m.shape == (12*nside_out**2,)
        assert m.dtype.metadata == {'spin': 0, 'kernel': 'healpix', 'power': 0}
        assert np.isclose(m.mean(), fsky)


def test_map_positions(nside, catalog, vmap):

    from le3_pk_wl.maps import map_positions

    # normal mode: compute overdensity maps with metadata

    m = map_positions(nside, catalog)

    assert m.shape == (12*nside**2,)
    assert m.dtype.metadata == {'spin': 0, 'nbar': 4., 'kernel': 'healpix', 'power': 0}
    np.testing.assert_array_equal(m, 0)

    # compute number count map

    m = map_positions(nside, catalog, overdensity=False)

    assert m.shape == (12*nside**2,)
    assert m.dtype.metadata == {'spin': 0, 'nbar': 4., 'kernel': 'healpix', 'power': 1}
    np.testing.assert_array_equal(m, 4)

    # compute overdensity maps with visibility map

    m = map_positions(nside, catalog, vmap)

    assert m.shape == (12*nside**2,)
    assert m.dtype.metadata == {'spin': 0, 'nbar': 4./vmap.mean(), 'kernel': 'healpix', 'power': 0}

    # compute number count map with visibility map

    m = map_positions(nside, catalog, vmap, overdensity=False)

    assert m.shape == (12*nside**2,)
    assert m.dtype.metadata == {'spin': 0, 'nbar': 4./vmap.mean(), 'kernel': 'healpix', 'power': 1}


def test_map_shears(nside, catalog):

    from le3_pk_wl.maps import map_shears

    m = map_shears(nside, catalog)

    w = next(iter(catalog)).w
    w = w.reshape(w.size//4, 4).sum(axis=-1)
    wbar = w.mean()

    assert m.shape == (2, 12*nside**2,)
    assert m.dtype.metadata == {'spin': 2, 'wbar': wbar, 'kernel': 'healpix', 'power': 0}
    np.testing.assert_array_almost_equal(m, 0)

    m = map_shears(nside, catalog, normalize=False)

    assert m.shape == (2, 12*nside**2,)
    assert m.dtype.metadata == {'spin': 2, 'wbar': wbar, 'kernel': 'healpix', 'power': 1}
    np.testing.assert_array_almost_equal(m, 0)


def test_map_weights(nside, catalog):

    from le3_pk_wl.maps import map_weights

    m = map_weights(nside, catalog)

    w = next(iter(catalog)).w
    w = w.reshape(w.size//4, 4).sum(axis=-1)
    wbar = w.mean()

    assert m.shape == (12*nside**2,)
    assert m.dtype.metadata == {'spin': 0, 'wbar': wbar, 'kernel': 'healpix', 'power': 0}
    np.testing.assert_array_almost_equal(m, w/wbar)

    m = map_weights(nside, catalog, normalize=False)

    assert m.shape == (12*nside**2,)
    assert m.dtype.metadata == {'spin': 0, 'wbar': wbar, 'kernel': 'healpix', 'power': 1}
    np.testing.assert_array_almost_equal(m, w)


def test_transform_maps():

    from le3_pk_wl.maps import transform_maps, update_metadata

    nside = 32
    npix = 12*nside**2

    t = np.random.randn(npix)
    update_metadata(t, spin=0, a=1)
    p = np.random.randn(2, npix)
    update_metadata(p, spin=2, b=2)

    # single scalar map
    maps = {('T', 0): t}
    alms = transform_maps(maps)

    assert len(alms) == 1
    assert alms.keys() == maps.keys()
    assert alms['T', 0].dtype.metadata['spin'] == 0
    assert alms['T', 0].dtype.metadata['a'] == 1
    assert alms['T', 0].dtype.metadata['nside'] == nside

    # polarisation map
    maps = {('P', 0): p}
    alms = transform_maps(maps)

    assert len(alms) == 2
    assert alms.keys() == {('E', 0), ('B', 0)}
    assert alms['E', 0].dtype.metadata['spin'] == 2
    assert alms['B', 0].dtype.metadata['spin'] == 2
    assert alms['E', 0].dtype.metadata['b'] == 2
    assert alms['B', 0].dtype.metadata['b'] == 2
    assert alms['E', 0].dtype.metadata['nside'] == nside
    assert alms['B', 0].dtype.metadata['nside'] == nside

    # mixed
    maps = {('T', 0): t, ('P', 1): p}
    alms = transform_maps(maps)

    assert len(alms) == 3
    assert alms.keys() == {('T', 0), ('E', 1), ('B', 1)}
    assert alms['T', 0].dtype.metadata['spin'] == 0
    assert alms['E', 1].dtype.metadata['spin'] == 2
    assert alms['B', 1].dtype.metadata['spin'] == 2
    assert alms['T', 0].dtype.metadata['a'] == 1
    assert alms['E', 1].dtype.metadata['b'] == 2
    assert alms['B', 1].dtype.metadata['b'] == 2
    assert alms['T', 0].dtype.metadata['nside'] == nside
    assert alms['E', 1].dtype.metadata['nside'] == nside
    assert alms['B', 1].dtype.metadata['nside'] == nside


def test_update_metadata():
    from le3_pk_wl.maps import update_metadata

    a = np.empty(0)

    assert a.dtype.metadata is None

    update_metadata(a, x=1)

    assert a.dtype.metadata == {'x': 1}

    update_metadata(a, y=2)

    assert a.dtype.metadata == {'x': 1, 'y': 2}


def test_map_catalogs(nside, catalog):

    from le3_pk_wl.maps import map_catalogs

    vmap = np.ones(12*nside**2)

    catalogs = {
        0: catalog,
        1: catalog,
    }

    with pytest.raises(KeyError):
        map_catalogs('v', nside, catalogs)

    vmaps = {
        0: vmap,
    }

    with pytest.raises(KeyError):
        map_catalogs('v', nside, catalogs)

    vmaps[None] = vmap

    for which in '', 'p', 'g', 'w', 'v', 'pg', 'pw', 'pv', 'gw', 'gv', 'wv', 'pgw', 'pgv', 'pwv', 'gwv', 'pgwv':

        maps = map_catalogs(which, nside, catalogs, vmaps)

        assert len(maps) == len(catalogs)*len(which)
        for i in catalogs:
            for k in map(str.upper, which):
                assert (k, i) in maps
