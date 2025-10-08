import numpy as np
import pytest
from pathlib import Path
from numba import config
import heracles
import healpy as hp
from heracles.healpy import HealpixMapper
from heracles.fields import Positions, Shears, Visibility, Weights

config.DISABLE_JIT = True


@pytest.fixture(scope="session")
def rng(seed: int = 50) -> np.random.Generator:
    return np.random.default_rng(seed)


@pytest.fixture(scope="session")
def data_maps():
    nbins = 2
    nside = 128
    lmax = 128
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
def vis_maps():
    nbins = 2
    nside = 128
    npix = hp.nside2npix(nside)
    map = 4 * np.ones(npix)
    maps = {}
    heracles.update_metadata(
        map,
        nside=nside,
        lmax=nside,
        bias=0.0,
        fsky=1 / 2,
        spin=0,
    )
    for i in range(1, nbins + 1):
        maps[("VIS", i)] = map
        maps[("WHT", i)] = np.array([map])
    return maps


@pytest.fixture(scope="session")
def fields():
    """
    Internal method to initialize fields.
    inputs:
        nside (int): Healpix nside
        lmax (int): Maximum multipole
    returns:
        fields (dict): Dictionary of fields
    """
    nside = 128
    lmax = 128
    mapper = HealpixMapper(nside=nside, lmax=lmax)
    fields = {
        "POS": Positions(mapper, mask="VIS"),
        "SHE": Shears(mapper, mask="WHT"),
        "VIS": Visibility(mapper),
        "WHT": Weights(mapper),
    }
    return fields


@pytest.fixture(scope="session")
def jk_maps():
    path = Path(__file__).parent / "data"
    nbins = 2
    jk_maps = {}
    for i in range(1, nbins + 1):
        jk_maps[("VIS", i)] = hp.read_map(path / "jkmap.fits")
        jk_maps[("WHT", i)] = hp.read_map(path / "jkmap.fits")
    return jk_maps
