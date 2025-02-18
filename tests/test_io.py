import pytest

import heracles


@pytest.fixture
def zbins():
    return {0: (0.0, 0.8), 1: (1.0, 1.2)}


@pytest.fixture
def mock_alms(rng, zbins):
    import numpy as np

    lmax = 32

    Nlm = (lmax + 1) * (lmax + 2) // 2

    # names and spins
    fields = {"P": 0, "G": 2}

    alms = {}
    for n, s in fields.items():
        shape = (Nlm, 2) if s == 0 else (2, Nlm, 2)
        for i in zbins:
            a = rng.standard_normal(shape) @ [1, 1j]
            a.dtype = np.dtype(a.dtype, metadata={"nside": 3, "spin": s})
            alms[n, i] = a

    return alms


@pytest.fixture
def mock_cls(rng):
    import numpy as np

    cl00 = rng.random(101)
    cl00.dtype = np.dtype(
        cl00.dtype,
        metadata={
            "catalog_1": "cat-a.fits",
            "nside_1": 32,
            "spin_1": 0,
            "catalog_2": "cat-b.fits",
            "nside_2": 64,
            "spin_2": 0,
        },
    )

    cl02 = rng.random((2, 101))
    cl02.dtype = np.dtype(
        cl02.dtype,
        metadata={
            "catalog_1": "cat-a.fits",
            "nside_1": 32,
            "spin_1": 0,
            "catalog_2": "cat-b.fits",
            "nside_2": 64,
            "spin_2": 2,
        },
    )

    cl22 = rng.random((3, 101))
    cl22.dtype = np.dtype(
        cl22.dtype,
        metadata={
            "catalog_1": "cat-a.fits",
            "nside_1": 32,
            "spin_1": 2,
            "catalog_2": "cat-b.fits",
            "nside_2": 64,
            "spin_2": 2,
        },
    )

    cl22x = rng.random((4, 101))
    cl22x.dtype = np.dtype(
        cl22x.dtype,
        metadata={
            "catalog_1": "cat-a.fits",
            "nside_1": 32,
            "spin_1": 2,
            "catalog_2": "cat-b.fits",
            "nside_2": 64,
            "spin_2": 2,
        },
    )

    return {
        ("POS", "POS", 0, 0): cl00,
        ("POS", "SHE", 0, 0): cl02,
        ("SHE", "SHE", 0, 0): cl22,
        ("POS", "POS", 0, 1): cl00,
        ("POS", "SHE", 0, 1): cl02,
        ("SHE", "SHE", 0, 1): cl22x,
        ("POS", "SHE", 1, 0): cl02,
        ("POS", "POS", 1, 1): cl00,
        ("POS", "SHE", 1, 1): cl02,
        ("SHE", "SHE", 1, 1): cl22,
    }


@pytest.fixture(scope="session")
def nside():
    return 32


@pytest.fixture(scope="session")
def datadir(tmp_path_factory):
    return tmp_path_factory.mktemp("data")


@pytest.fixture(scope="session")
def mock_vmap_fields(nside, rng):
    import healpy as hp
    import numpy as np

    nfields = 4

    npix = hp.nside2npix(nside)
    maps = rng.random(npix * nfields).reshape((npix, nfields))
    pixels = np.unique(rng.integers(0, npix, size=npix // 3))
    vmappix = np.delete(np.arange(0, npix), pixels)
    for i in range(nfields):
        maps[:, i][vmappix] = 0
    return [maps, pixels]


@pytest.fixture(scope="session")
def mock_vmap_partial(mock_vmap_fields, nside, datadir):
    import fitsio

    filename = str(datadir / "vmap_partial.fits")

    maps, pixels = mock_vmap_fields

    fits = fitsio.FITS(filename, "rw")
    fits.write(data=None)
    fits.write_table(
        [
            pixels,
            maps[:, 0][pixels],
            maps[:, 1][pixels],
            maps[:, 2][pixels],
            maps[:, 3][pixels],
        ],
        names=["PIXEL", "WEIGHT", "FIELD1", "FIELD2", "FIELD3"],
        header={
            "NSIDE": nside,
            "ORDERING": "RING",
            "INDXSCHM": "EXPLICIT",
            "OBJECT": "PARTIAL",
        },
    )
    fits.close()

    return filename


@pytest.fixture(scope="session")
def mock_vmap(mock_vmap_fields, nside, datadir):
    import fitsio

    filename = str(datadir / "vmap.fits")

    maps, _ = mock_vmap_fields

    fits = fitsio.FITS(filename, "rw")
    fits.write(data=None)
    fits.write_table(
        [maps[:, 0], maps[:, 1], maps[:, 2], maps[:, 3]],
        names=["WEIGHT", "FIELD1", "FIELD2", "FIELD3"],
        header={
            "NSIDE": nside,
            "ORDERING": "RING",
            "INDXSCHM": "IMPLICIT",
            "OBJECT": "FULLSKY",
        },
    )
    fits.close()

    return filename


def test_write_read_maps(rng, tmp_path):
    import healpy as hp
    import numpy as np

    from heracles.io import read_maps, write_maps

    nside = 4
    npix = 12 * nside**2

    p = rng.random(npix)
    v = rng.random(npix)
    g = rng.random((2, npix))

    p.dtype = np.dtype(p.dtype, metadata={"catalog": "cat.fits", "spin": 0})
    v.dtype = np.dtype(v.dtype, metadata={"catalog": "cat.fits", "spin": 0})
    g.dtype = np.dtype(g.dtype, metadata={"catalog": "cat.fits", "spin": 2})

    maps = {
        ("P", 1): p,
        ("V", 2): v,
        ("G", 3): g,
    }

    write_maps("maps.fits", maps, workdir=str(tmp_path))
    assert (tmp_path / "maps.fits").exists()
    maps_r = read_maps("maps.fits", workdir=str(tmp_path))

    assert maps.keys() == maps_r.keys()
    for key in maps:
        np.testing.assert_array_equal(maps[key], maps_r[key])
        assert maps[key].dtype.metadata == maps_r[key].dtype.metadata

    # make sure map can be read by healpy
    m = hp.read_map(tmp_path / "maps.fits", hdu="MAP0")
    np.testing.assert_array_equal(maps["P", 1], m)


def test_write_read_alms(mock_alms, tmp_path):
    import numpy as np

    from heracles.io import read_alms, write_alms

    write_alms("alms.fits", mock_alms, workdir=str(tmp_path))
    assert (tmp_path / "alms.fits").exists()
    alms = read_alms("alms.fits", workdir=str(tmp_path))

    assert alms.keys() == mock_alms.keys()
    for key in mock_alms:
        np.testing.assert_array_equal(mock_alms[key], alms[key])
        assert mock_alms[key].dtype.metadata == alms[key].dtype.metadata


def test_write_read_cls(mock_cls, tmp_path):
    import numpy as np

    path = tmp_path / "test.fits"

    assert not path.exists()

    heracles.write(path, mock_cls)

    assert path.exists()

    cls = heracles.read(path)

    assert list(cls.keys()) == list(mock_cls.keys())
    for key in mock_cls:
        assert key in cls
        cl, mock_cl = cls[key], mock_cls[key]
        assert cl.axis == (cl.ndim - 1,)
        lmax = mock_cl.shape[-1] - 1
        np.testing.assert_array_equal(cl, mock_cl)
        np.testing.assert_array_equal(cl.ell, np.arange(lmax + 1))
        assert cl.dtype.metadata == mock_cl.dtype.metadata


def test_write_read_mms(rng, tmp_path):
    import numpy as np

    path = tmp_path / "test.fits"

    mms = {
        ("POS", "POS", 0, 1): heracles.Result(rng.standard_normal((10, 10)), axis=0),
        ("POS", "SHE", 1, 2): heracles.Result(rng.standard_normal((20, 5)), axis=0),
        ("SHE", "SHE", 2, 3): heracles.Result(rng.standard_normal((10, 5, 2)), axis=1),
    }

    assert not path.exists()

    heracles.write(path, mms)

    assert path.exists()

    mms_ = heracles.read(path)

    assert mms_.keys() == mms.keys()
    for key in mms:
        assert key in mms_
        mm = mms_[key]
        assert mm.axis == (mm.ndim - 2,)
        lmax = mm.shape[-2] - 1
        np.testing.assert_array_equal(mm, mms[key])
        np.testing.assert_array_equal(mm.ell, np.arange(lmax + 1))


def test_write_read_cov(mock_cls, tmp_path):
    from itertools import combinations_with_replacement

    import numpy as np

    cov = {}
    for (a1, b1, i1, j1), (a2, b2, i2, j2) in combinations_with_replacement(
        mock_cls, 2
    ):
        cl1, cl2 = mock_cls[a1, b1, i1, j1], mock_cls[a2, b2, i2, j2]
        c = cl1[..., None, : cl1.shape[-1] // 2, None] * cl2[..., None, :]
        cov[a1, b1, a2, b2, i1, j1, i2, j2] = heracles.Result(c, axis=(-2, -1))

    path = tmp_path / "cov.fits"

    heracles.write(path, cov)

    assert path.exists()

    cov_ = heracles.read(path)

    assert cov_.keys() == cov.keys()
    for key in cov:
        assert key in cov_
        c = cov_[key]
        assert c.axis == (c.ndim - 2, c.ndim - 1)
        lmax1 = c.shape[-2] - 1
        lmax2 = c.shape[-1] - 1
        np.testing.assert_array_equal(c, cov[key])
        np.testing.assert_array_equal(c.ell[0], np.arange(lmax1 + 1))
        np.testing.assert_array_equal(c.ell[1], np.arange(lmax2 + 1))


def test_read_vmap_partial(mock_vmap_fields, mock_vmap_partial, nside):
    import healpy as hp

    from heracles.io import read_vmap

    maps = mock_vmap_fields[0]

    vmap = read_vmap(mock_vmap_partial, nside=nside)
    assert (vmap == maps[:, 0]).all()

    field = 2
    vmap = read_vmap(mock_vmap_partial, nside=nside, field=field)
    assert (vmap == maps[:, field]).all()

    field = 3
    with pytest.warns():
        vmap = read_vmap(mock_vmap_partial, nside=nside // 2, field=field)
    vmapud = hp.pixelfunc.ud_grade(maps[:, field], nside // 2)
    assert (vmap == vmapud).all()


def test_read_vmap(mock_vmap_fields, mock_vmap, nside):
    import healpy as hp

    from heracles.io import read_vmap

    maps = mock_vmap_fields[0]

    vmap = read_vmap(mock_vmap, nside=nside)
    assert (vmap == maps[:, 0]).all()

    fields = 2
    vmap = read_vmap(mock_vmap, nside=nside, field=fields)
    assert (vmap == maps[:, fields]).all()

    fields = 3
    with pytest.warns():
        vmap = read_vmap(mock_vmap, nside=nside // 2, field=fields)
    vmapud = hp.pixelfunc.ud_grade(maps[:, fields], nside // 2)
    assert (vmap == vmapud).all()

    fields = 3
    with pytest.warns():
        vmap = read_vmap(mock_vmap, nside=nside * 2, field=fields)
    vmapud = hp.pixelfunc.ud_grade(maps[:, fields], nside * 2)
    assert (vmap == vmapud).all()


def test_tocfits(tmp_path):
    import fitsio
    import numpy as np

    from heracles.io import TocFits

    class TestFits(TocFits):
        tag = "TEST"

    path = tmp_path / "test.fits"

    assert not path.exists()

    tocfits = TestFits(path, clobber=True)

    assert path.exists()

    with fitsio.FITS(path) as fits:
        assert len(fits) == 1

    assert len(tocfits) == 0
    assert list(tocfits) == []
    assert tocfits.toc == {}

    data12 = np.zeros(5, dtype=[("X", float), ("Y", int)])
    data22 = np.ones(5, dtype=[("X", float), ("Y", int)])
    data21 = np.full(5, 2, dtype=[("X", float), ("Y", int)])

    tocfits[1, 2] = data12

    with fitsio.FITS(path) as fits:
        assert len(fits) == 2
        np.testing.assert_array_equal(fits["TEST0"].read(), data12)

    assert len(tocfits) == 1
    assert list(tocfits) == [(1, 2)]
    assert tocfits.toc == {(1, 2): "TEST0"}
    np.testing.assert_array_equal(tocfits[1, 2], data12)

    tocfits[2, 2] = data22

    with fitsio.FITS(path) as fits:
        assert len(fits) == 3
        np.testing.assert_array_equal(fits["TEST0"].read(), data12)
        np.testing.assert_array_equal(fits["TEST1"].read(), data22)

    assert len(tocfits) == 2
    assert list(tocfits) == [(1, 2), (2, 2)]
    assert tocfits.toc == {(1, 2): "TEST0", (2, 2): "TEST1"}
    np.testing.assert_array_equal(tocfits[1, 2], data12)
    np.testing.assert_array_equal(tocfits[2, 2], data22)

    with pytest.raises(NotImplementedError):
        del tocfits[1, 2]

    del tocfits

    tocfits2 = TestFits(path, clobber=False)

    assert len(tocfits2) == 2
    assert list(tocfits2) == [(1, 2), (2, 2)]
    assert tocfits2.toc == {(1, 2): "TEST0", (2, 2): "TEST1"}
    np.testing.assert_array_equal(tocfits2[1, 2], data12)
    np.testing.assert_array_equal(tocfits2[2, 2], data22)

    tocfits2[2, 1] = data21

    with fitsio.FITS(path) as fits:
        assert len(fits) == 4
        np.testing.assert_array_equal(fits["TEST0"].read(), data12)
        np.testing.assert_array_equal(fits["TEST1"].read(), data22)
        np.testing.assert_array_equal(fits["TEST2"].read(), data21)


def test_tocfits_is_lazy(tmp_path):
    import fitsio

    from heracles.io import TocFits

    path = tmp_path / "bad.fits"

    # test keys(), values(), and items() are not eagerly reading data
    tocfits = TocFits(path, clobber=True)

    # manually enter some non-existent rows into the ToC
    assert tocfits._toc == {}
    tocfits._toc[0,] = "BAD0"
    tocfits._toc[1,] = "BAD1"
    tocfits._toc[2,] = "BAD2"

    # these should not error
    tocfits.keys()
    tocfits.values()
    tocfits.items()

    # contains and iteration are lazy
    assert 0 in tocfits
    assert list(tocfits) == [(0,), (1,), (2,)]

    # subselection should work fine
    selected = tocfits[...]
    assert isinstance(selected, TocFits)
    assert len(selected) == 3

    # make sure nothing is in the FITS
    with fitsio.FITS(path, "r") as fits:
        assert len(fits) == 1

    # make sure there are errors when acualising the generators
    with pytest.raises(OSError):
        list(tocfits.values())
