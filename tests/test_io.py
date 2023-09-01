import pytest

NFIELDS_TEST = 4


@pytest.fixture
def zbins():
    return {0: (0.0, 0.8), 1: (1.0, 1.2)}


@pytest.fixture
def mock_alms(zbins):
    import numpy as np

    lmax = 32

    Nlm = (lmax + 1) * (lmax + 2) // 2

    names = ["P", "E", "B"]

    alms = {}
    for n in names:
        for i in zbins:
            a = np.random.randn(Nlm, 2) @ [1, 1j]
            a.dtype = np.dtype(a.dtype, metadata={"nside": 32})
            alms[n, i] = a

    return alms


@pytest.fixture
def mock_cls():
    import numpy as np

    cl = np.random.rand(101)
    cl.dtype = np.dtype(cl.dtype, metadata={"nside_1": 32, "nside_2": 64})

    return {
        ("P", "P", 0, 0): cl,
        ("P", "G_E", 0, 0): cl,
        ("P", "G_B", 0, 0): cl,
        ("G_E", "G_E", 0, 0): cl,
        ("G_B", "G_B", 0, 0): cl,
        ("G_E", "G_B", 0, 0): cl,
        ("P", "P", 0, 1): cl,
        ("P", "G_E", 0, 1): cl,
        ("P", "G_B", 0, 1): cl,
        ("G_E", "G_E", 0, 1): cl,
        ("G_B", "G_B", 0, 1): cl,
        ("G_E", "G_B", 0, 1): cl,
        ("P", "G_E", 1, 0): cl,
        ("P", "G_B", 1, 0): cl,
        ("G_E", "G_B", 1, 0): cl,
        ("P", "P", 1, 1): cl,
        ("P", "G_E", 1, 1): cl,
        ("P", "G_B", 1, 1): cl,
        ("G_E", "G_E", 1, 1): cl,
        ("G_B", "G_B", 1, 1): cl,
        ("G_E", "G_B", 1, 1): cl,
    }


@pytest.fixture(scope="session")
def nside():
    return 32


@pytest.fixture(scope="session")
def datadir(tmp_path_factory):
    return tmp_path_factory.mktemp("data")


@pytest.fixture(scope="session")
def mock_mask_fields(nside):
    import healpy as hp
    import numpy as np

    npix = hp.nside2npix(nside)
    maps = np.random.rand(npix * NFIELDS_TEST).reshape((npix, NFIELDS_TEST))
    pixels = np.unique(np.random.randint(0, npix, npix // 3))
    maskpix = np.delete(np.arange(0, npix), pixels)
    for i in range(NFIELDS_TEST):
        maps[:, i][maskpix] = 0
    return [maps, pixels]


@pytest.fixture(scope="session")
def mock_writemask_partial(mock_mask_fields, nside, datadir):
    import fitsio

    filename = str(datadir / "mask_partial.fits")

    maps, pixels = mock_mask_fields

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
def mock_writemask_full(mock_mask_fields, nside, datadir):
    import fitsio

    filename = str(datadir / "mask_full.fits")

    maps, _ = mock_mask_fields

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


@pytest.fixture(scope="session")
def mock_mask_extra(nside):
    import healpy as hp
    import numpy as np

    npix = hp.nside2npix(nside)
    maps = np.random.rand(npix)
    pixels = np.unique(np.random.randint(0, npix, npix // 3))
    maskpix = np.delete(np.arange(0, npix), pixels)
    maps[maskpix] = 0
    return [maps, pixels]


@pytest.fixture(scope="session")
def mock_writemask_extra(mock_mask_extra, nside, datadir):
    import fitsio

    filename = str(datadir / "mask_extra.fits")

    maps, _ = mock_mask_extra

    fits = fitsio.FITS(filename, "rw")
    fits.write(data=None)
    fits.write_table(
        [maps],
        names=["WEIGHT"],
        header={
            "NSIDE": nside,
            "ORDERING": "RING",
            "INDXSCHM": "IMPLICIT",
            "OBJECT": "FULLSKY",
        },
    )
    fits.close()

    return filename


def test_write_read_maps(tmp_path):
    import healpy as hp
    import numpy as np

    from heracles.io import read_maps, write_maps

    nside = 4
    npix = 12 * nside**2

    p = np.random.rand(npix)
    v = np.random.rand(npix)
    g = np.random.rand(2, npix)

    p.dtype = np.dtype(p.dtype, metadata={"spin": 0})
    v.dtype = np.dtype(v.dtype, metadata={"spin": 0})
    g.dtype = np.dtype(g.dtype, metadata={"spin": 0})

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

    from heracles.io import read_cls, write_cls

    filename = "test.fits"
    workdir = str(tmp_path)

    write_cls(filename, mock_cls, workdir=workdir)

    assert (tmp_path / filename).exists()

    cls = read_cls(filename, workdir=workdir)

    assert cls.keys() == mock_cls.keys()
    for key in mock_cls:
        assert key in cls
        cl = cls[key]
        assert cl.dtype.names == ("L", "CL", "LMIN", "LMAX", "W")
        np.testing.assert_array_equal(cl["L"], np.arange(len(mock_cls[key])))
        np.testing.assert_array_equal(cl["CL"], mock_cls[key])
        assert cl.dtype.metadata == mock_cls[key].dtype.metadata


def test_write_read_mms(tmp_path):
    import numpy as np

    from heracles.io import read_mms, write_mms

    filename = "test.fits"
    workdir = str(tmp_path)

    mms = {
        ("00", 0, 1): np.random.randn(10, 10),
        ("0+", 1, 2): np.random.randn(20, 5),
        ("++", 2, 3): np.random.randn(10, 5, 2),
    }

    write_mms(filename, mms, workdir=workdir)

    assert (tmp_path / filename).exists()

    mms_ = read_mms(filename, workdir=workdir)

    assert mms_.keys() == mms.keys()
    for key in mms:
        np.testing.assert_array_equal(mms_[key], mms[key])


def test_write_read_cov(mock_cls, tmp_path):
    from itertools import combinations_with_replacement

    import numpy as np

    from heracles.io import read_cov, write_cov

    workdir = str(tmp_path)

    cov = {}
    for k1, k2 in combinations_with_replacement(mock_cls, 2):
        cl1, cl2 = mock_cls[k1], mock_cls[k2]
        cov[k1, k2] = np.outer(cl1, cl2)

    filename = "cov.fits"

    write_cov(filename, cov, workdir=workdir)

    assert (tmp_path / filename).exists()

    cov_ = read_cov(filename, workdir=workdir)

    assert cov_.keys() == cov.keys()
    for key in cov:
        np.testing.assert_array_equal(cov_[key], cov[key])


def test_read_mask_partial(mock_mask_fields, mock_writemask_partial, nside):
    import healpy as hp

    from heracles.io import read_mask

    maps = mock_mask_fields[0]

    mask = read_mask(mock_writemask_partial, nside=nside)
    assert (mask == maps[:, 0]).all()

    ibin = 2
    mask = read_mask(mock_writemask_partial, nside=nside, field=ibin)
    assert (mask == maps[:, ibin]).all()

    ibin = 3
    mask = read_mask(mock_writemask_partial, nside=nside // 2, field=ibin)
    maskud = hp.pixelfunc.ud_grade(maps[:, ibin], nside // 2)
    assert (mask == maskud).all()


def test_read_mask_full(mock_mask_fields, mock_writemask_full, nside):
    import healpy as hp

    from heracles.io import read_mask

    maps = mock_mask_fields[0]

    mask = read_mask(mock_writemask_full, nside=nside)
    assert (mask == maps[:, 0]).all()

    ibin = 2
    mask = read_mask(mock_writemask_full, nside=nside, field=ibin)
    assert (mask == maps[:, ibin]).all()

    ibin = 3
    mask = read_mask(mock_writemask_full, nside=nside // 2, field=ibin)
    maskud = hp.pixelfunc.ud_grade(maps[:, ibin], nside // 2)
    assert (mask == maskud).all()

    ibin = 3
    mask = read_mask(mock_writemask_full, nside=nside * 2, field=ibin)
    maskud = hp.pixelfunc.ud_grade(maps[:, ibin], nside * 2)
    assert (mask == maskud).all()


def test_read_mask_extra(
    mock_mask_fields,
    mock_mask_extra,
    mock_writemask_full,
    nside,
    mock_writemask_extra,
):
    from heracles.io import read_mask

    maps = mock_mask_fields[0]
    maps_extra = mock_mask_extra[0]

    ibin = 2
    mask = read_mask(
        mock_writemask_full,
        nside=nside,
        field=ibin,
        extra_mask_name=mock_writemask_extra,
    )
    assert (mask == maps[:, ibin] * maps_extra[:]).all()
