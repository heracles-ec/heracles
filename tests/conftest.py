import numpy as np
import pytest
# from numba import config


# config.DISABLE_JIT = True


@pytest.fixture(scope="session", autouse=True)
def nside():
    return 32


@pytest.fixture(scope="session", autouse=True)
def njk():
    return 3


@pytest.fixture(scope="session")
def rng(seed: int = 50) -> np.random.Generator:
    return np.random.default_rng(seed)


@pytest.fixture(scope="session")
def data_maps(nside):
    import healpy as hp
    import heracles

    nbins = 2
    lmax = nside // 4
    npix = hp.nside2npix(nside)
    fsky = 1 / 2
    ngal = 4.0
    wmean = 1.0
    var = 1.0
    bias = 4 * np.pi * fsky**2 * (var / wmean**2) / ngal
    map_p = 4 * np.ones(npix)
    nbar = (ngal * wmean) / (fsky * npix)
    heracles.update_metadata(
        map_p,
        nside=nside,
        lmax=lmax,
        ngal=ngal,
        nbar=nbar,
        wmean=wmean,
        bias=bias,
        var=var,
        variance=var / wmean**2,
        neff=ngal / (4 * np.pi * fsky),
        fsky=fsky,
        spin=0,
    )
    map_g = 4 * np.ones(npix)
    heracles.update_metadata(
        map_g,
        nside=nside,
        lmax=lmax,
        ngal=ngal,
        nbar=nbar,
        wmean=wmean,
        bias=bias,
        var=var,
        variance=var / wmean**2,
        neff=ngal / (2 * np.pi * fsky),
        fsky=fsky,
        spin=2,
    )
    maps = {}
    for i in range(1, nbins + 1):
        maps[("POS", i)] = map_p
        maps[("SHE", i)] = np.array([map_g, map_g])
    return maps


@pytest.fixture(scope="session")
def vis_maps(nside):
    import healpy as hp
    import heracles

    nbins = 2
    npix = hp.nside2npix(nside)
    map = 4 * np.ones(npix)
    maps = {}
    heracles.update_metadata(
        map,
        nside=nside,
        lmax=nside // 4,
        bias=0.0,
        fsky=1 / 2,
        spin=0,
    )
    for i in range(1, nbins + 1):
        maps[("VIS", i)] = map
        maps[("WHT", i)] = np.array([map])
    return maps


@pytest.fixture(scope="session")
def fields(nside):
    """
    Internal method to initialize fields.
    inputs:
        nside (int): Healpix nside
        lmax (int): Maximum multipole
    returns:
        fields (dict): Dictionary of fields
    """
    from heracles.healpy import HealpixMapper
    from heracles.fields import Positions, Shears, Visibility, Weights

    lmax = nside // 4
    mapper = HealpixMapper(nside=nside, lmax=lmax)
    fields = {
        "POS": Positions(mapper, mask="VIS"),
        "SHE": Shears(mapper, mask="WHT"),
        "VIS": Visibility(mapper),
        "WHT": Weights(mapper),
    }
    return fields


@pytest.fixture(scope="session")
def jk_maps(nside, njk):
    npix = 12 * nside**2
    jkmap = np.ones(npix)
    segment = npix // njk
    for i in range(njk):
        jkmap[i * segment : (i + 1) * segment] = i + 1
    return {
        ("VIS", 1): jkmap,
        ("WHT", 1): jkmap,
        ("VIS", 2): jkmap,
        ("WHT", 2): jkmap,
    }


@pytest.fixture(scope="session")
def cls0(fields, data_maps, jk_maps):
    from heracles.dices.jackknife import get_cls

    return get_cls(data_maps, jk_maps, fields)


@pytest.fixture(scope="session")
def mls0(fields, vis_maps, jk_maps):
    from heracles.dices.jackknife import get_cls

    return get_cls(vis_maps, jk_maps, fields)


@pytest.fixture(scope="session")
def cls1(fields, data_maps, vis_maps, jk_maps):
    from heracles.dices.jackknife import jackknife_cls

    return jackknife_cls(data_maps, vis_maps, jk_maps, fields, nd=1)


@pytest.fixture(scope="session")
def cls2(fields, data_maps, vis_maps, jk_maps):
    from heracles.dices.jackknife import jackknife_cls

    return jackknife_cls(data_maps, vis_maps, jk_maps, fields, nd=2)


@pytest.fixture(scope="session")
def cov_jk(cls1):
    from heracles.dices import jackknife_covariance

    return jackknife_covariance(cls1)
