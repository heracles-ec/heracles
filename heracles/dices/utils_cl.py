# DICES: Euclid code for harmonic-space statistics on the sphere
#
# Copyright (C) 2023-2024 Euclid Science Ground Segment
#
# This file is part of DICES.
#
# DICES is free software: you can redistribute it and/or modify it
# under the terms of the GNU Lesser General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# DICES is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public
# License along with DICES. If not, see <https://www.gnu.org/licenses/>.
"""The DICES command line interface."""

import numpy as np
from ..result import Result
import logging
import time
from datetime import timedelta
from itertools import combinations_with_replacement

logger = logging.getLogger(__name__)


def get_lgrid(lmin, lmax, nbin, uselog=True):
    """
    Produces a grid of l values.
    input:
        lmin: minimum l value
        lmax: maximum l value
        nbin: number of bins
        uselog: use log spacing
    returns:
        lgrid: grid of l values
    """
    if uselog:
        ledges = np.logspace(np.log10(lmin), np.log10(lmax), nbin + 1)
    else:
        ledges = np.linspace(lmin, lmax, nbin + 1)
    lgrid = 0.5 * (ledges[1:] + ledges[:-1])
    dl = ledges[1:] - ledges[:-1]
    return lgrid, ledges, dl


def compsep_Cls(Cls):
    """
    Separates the SHE values into E and B modes.
    input:
        Cls: dictionary of Cl values
    returns:
        Cls_unraveled: dictionary of Cl values
    """
    Cls_compsep = {}
    for key in list(Cls.keys()):
        t1, t2, b1, b2 = key
        cl = Cls[key]
        ell = cl.ell
        if t1 == t2 == "POS":
            ndims = len(cl.shape)
            if ndims == 3:
                Cls_compsep[key] = Result(cl[..., 0, :], ell)
            else:
                Cls_compsep[key] = Result(cl[..., :], ell)
        elif t1 == t2 == "SHE" and b1 == b2:
            Cls_compsep[("G_E", "G_E", b1, b2)] = Result(cl[..., 0, :], ell)
            Cls_compsep[("G_B", "G_B", b1, b2)] = Result(cl[..., 1, :], ell)
            Cls_compsep[("G_E", "G_B", b1, b2)] = Result(cl[..., 2, :], ell)
        elif t1 == t2 == "SHE" and b1 != b2:
            Cls_compsep[("G_E", "G_E", b1, b2)] = Result(cl[..., 0, :], ell)
            Cls_compsep[("G_B", "G_B", b1, b2)] = Result(cl[..., 1, :], ell)
            Cls_compsep[("G_E", "G_B", b1, b2)] = Result(cl[..., 2, :], ell)
            Cls_compsep[("G_E", "G_B", b2, b1)] = Result(cl[..., 3, :], ell)
        elif t1 == "POS" and t2 == "SHE":
            Cls_compsep[("POS", "G_E", b1, b2)] = Result(cl[..., 0, :], ell)
            Cls_compsep[("POS", "G_B", b1, b2)] = Result(cl[..., 1, :], ell)
    return Cls_compsep


def get_Clkey(sources1, sources2):
    """
    Produces a Cl key for data maps.
    input:
        sources1: list of sources
        sources2: list of sources
    returns:
        Clkey: Cl key
    """
    n1, i1 = sources1
    n2, i2 = sources2
    if n2 == "POS" and n1 != "POS":
        Clkey = (n2, n1, i2, i1)
    elif n1 == "G_B" and n2 == "G_E":
        Clkey = (n2, n1, i2, i1)
    else:
        if n1 == n2 and i1 > i2:
            Clkey = (n1, n2, i2, i1)
        else:
            Clkey = (n1, n2, i1, i2)
    return Clkey


def add_to_Cls(Cls, x):
    """
    Adds a dictionary of Cl values to another.
    input:
        Cls: dictionary of Cl values
        x: dictionary of Cl values
    returns:
        Cls: updated dictionary of Cl values
    """
    _Cls = {}
    for key in Cls.keys():
        ell = Cls[key].ell
        _Cls[key] = Result(Cls[key].__array__() + x[key], ell)
    return _Cls


def sub_to_Cls(Cls, x):
    """
    Substracts a dictionary of Cl values to another.
    input:
        Cls: dictionary of Cl values
        x: dictionary of Cl values
    returns:
        Cls: updated dictionary of Cl values
    """
    _Cls = {}
    for key in Cls.keys():
        ell = Cls[key].ell
        _Cls[key] = Result(Cls[key].__array__() - x[key], ell)
    return _Cls


# Generic covariance calculation


def update_metadata(array, **metadata):
    """
    Update the metadata of an array.
    input:
        array: array
        metadata: dictionary of metadata
    """
    md = {}
    if array.dtype.metadata is not None:
        md.update(array.dtype.metadata)
    md.update(metadata)
    # create the new dtype with only the new metadata
    dt = array.dtype
    if dt.fields is not None:
        dt = dt.fields
    else:
        dt = dt.str
    dt = np.dtype(dt, metadata=md)
    # check that new dtype is compatible with old one
    if not np.can_cast(dt, array.dtype, casting="no"):
        raise ValueError("array with unsupported dtype")
    # set the new dtype in array
    array.dtype = dt


def cov_zero(nrows, ncols=None):
    """
    Produces a zero covariance matrix.
    input:
        nrows: number of rows
        ncols: number of columns
    returns:
        cov: covariance matrix
    """
    if ncols is None:
        ncols = nrows
    cov = np.zeros((nrows, ncols))
    rmu = np.zeros(nrows)
    cmu = np.zeros(ncols)
    update_metadata(cov, sample_count=0, sample_row_mean=rmu, sample_col_mean=cmu)
    return cov


def cov_add(cov, x, y=None):
    """
    Adds a sample to a sample covariance matrix.
    input:
        cov: covariance matrix
        x: sample
        y: sample
    returns:
        cov: updated covariance matrix
    """
    x = np.reshape(x, -1)
    if y is None:
        y = x
    else:
        y = np.reshape(y, -1)
    md = cov.dtype.metadata
    if not md:
        raise ValueError(
            "covariance matrix is missing metadata, "
            "use cov_zero() to initialise the array"
        )
    n = md["sample_count"]
    r = md["sample_row_mean"]
    c = md["sample_col_mean"]
    if x.size != r.size or y.size != c.size:
        raise ValueError("size mismatch between sample and covariance matrix")
    delta = x - r
    n += 1
    r += delta / n
    c += (y - c) / n
    if n > 1:
        cov += (np.outer(delta, y - c) - cov) / (n - 1)
    update_metadata(cov, sample_count=n)
    return cov


def update_covariance(cov, sample):
    """
    Updates a covariance matrix with a sample.
    input:
        cov: covariance matrix
        sample: dictionary of samples
    returns:
        cov: updated covariance matrix
    """
    logger.info("updating covariances for %d item(s)", len(sample))
    t = time.monotonic()
    for (k1, v1), (k2, v2) in combinations_with_replacement(sample.items(), 2):
        kk = (k1[0], k1[1], k2[0], k2[1], k1[2], k1[3], k2[2], k2[3])
        if kk not in cov:
            nrows, ncols = np.size(v1), np.size(v2)
            logger.info("creating %d x %d covariance matrix for %s", nrows, ncols, kk)
            cov[kk] = cov_zero(nrows, ncols)
        logger.info("updating covariance for %s", kk)
        cov[kk] = cov_add(cov[kk], v1, v2)
    logger.info(
        "updated %d covariance(s) in %s",
        len(sample) * (len(sample) + 1) // 2,
        timedelta(seconds=(time.monotonic() - t)),
    )
    return cov


def cov2corr(cov):
    """
    Produces a correlation matrix from a covariance matrix.
    input:
        cov: covariance matrix
    returns:
        corr: correlation matrix
    """
    corr = np.copy(cov)
    sig = np.sqrt(np.diag(cov))
    corr /= np.outer(sig, sig)
    return corr


def make_posdef(matrix):
    """
    Ensure the given symmetric matrix is positive definite.
    If not, adjust eigenvalues to make it positive definite.
    """
    if not np.allclose(matrix, matrix.T):
        print("Matrix is not symmetric.")
        matrix = 0.5 * (matrix + matrix.T)
    v, w = np.linalg.eigh(matrix)
    v[v < 0.0] = np.min(np.abs(v))
    return w @ np.diag(v) @ w.T


# Summary statistics for Cls
def get_Cl_mu(Clss):
    Cl_mu = {}
    total = len(Clss)
    for i, key in enumerate(list(Clss.keys())):
        Cls = Clss[key]
        if i == 0:
            for key in Cls.keys():
                cl = Cls[key]
                ell = cl.ell
                Cl_mu[key] = Result(cl, ell)
        else:
            for key in Cls.keys():
                cl = Cls[key].__array__() + Cl_mu[key].__array__()
                ell = Cls[key].ell
                Cl_mu[key] = Result(cl, ell)
    for key in Cl_mu.keys():
        cl = Cl_mu[key]
        ell = cl.ell
        Cl_mu[key] = Result(cl.__array__() / total, ell)
    return Cl_mu


def get_Cl_cov(Clss, jk=False):
    """
    Produces a covariance matrix from a list of Cl values.
    input:
        Clss: list of Cl values
        jk: use jackknife covariance
    returns:
        Cl_cov: covariance matrix
    """
    Cl_cov = {}
    for i, Cls in enumerate(Clss):
        Cl_cov = update_covariance(Cl_cov, Cls)
    if jk:
        numjk = len(Clss)
        jkfactor = (numjk - 1) * (numjk - 1) / numjk
        for key in Cl_cov.keys():
            Cl_cov[key] *= jkfactor
    return Cl_cov


def dict2mat(cls, cov):
    Clkeys = list(cls.keys())
    nells = [len(cls[key].ell) for key in Clkeys]
    full_cov = np.zeros((np.sum(nells), np.sum(nells)))
    for i in range(0, len(Clkeys)):
        for j in range(i, len(Clkeys)):
            ki = Clkeys[i]
            kj = Clkeys[j]
            A, B, nA, nB = ki[0], ki[1], ki[2], ki[3]
            C, D, nC, nD = kj[0], kj[1], kj[2], kj[3]
            covkey = (A, B, C, D, nA, nB, nC, nD)
            size_i = nells[i]
            size_j = nells[j]
            full_cov[i * size_i : (i + 1) * size_i, j * size_j : (j + 1) * size_j] = (
                cov[covkey]
            )
            if i != j:
                full_cov[
                    j * size_j : (j + 1) * size_j, i * size_i : (i + 1) * size_i
                ] = cov[covkey].T
    return full_cov


def mat2dict(cls, cov):
    Clkeys = list(cls.keys())
    nells = [len(cls[key].ell) for key in Clkeys]
    Cl_cov_dict = {}
    for i in range(0, len(Clkeys)):
        for j in range(i, len(Clkeys)):
            ki = Clkeys[i]
            kj = Clkeys[j]
            A, B, nA, nB = ki[0], ki[1], ki[2], ki[3]
            C, D, nC, nD = kj[0], kj[1], kj[2], kj[3]
            covkey = (A, B, C, D, nA, nB, nC, nD)
            size_i = nells[i]
            size_j = nells[j]
            Cl_cov_dict[covkey] = cov[
                i * size_i : (i + 1) * size_i, j * size_j : (j + 1) * size_j
            ]
            if i != j:
                Cl_cov_dict[covkey] = cov[
                    j * size_j : (j + 1) * size_j, i * size_i : (i + 1) * size_i
                ]
    return Cl_cov_dict


def split_comps(covkey, ncls1, ncls2):
    a1, b1, a2, b2, i1, j1, i2, j2 = covkey
    if ncls1 == 1:
        f1 = [("POS", "POS")]
    elif ncls1 == 2:
        f1 = [("POS", "G_E"), ("POS", "G_B")]
    elif ncls1 == 3:
        f1 = [("G_E", "G_E"), ("G_B", "G_B"), ("G_E", "G_B")]
    elif ncls2 == 4:
        f1 = [("G_E", "G_E"), ("G_B", "G_B"), ("G_E", "G_B"), ("G_B", "G_E")]

    if ncls2 == 1:
        f2 = [("POS", "POS")]
    elif ncls2 == 2:
        f2 = [("POS", "G_E"), ("POS", "G_B")]
    elif ncls2 == 3:
        f2 = [("G_E", "G_E"), ("G_B", "G_B"), ("G_E", "G_B")]
    elif ncls2 == 4:
        f2 = [("G_E", "G_E"), ("G_B", "G_B"), ("G_E", "G_B"), ("G_B", "G_E")]

    covkeys = {}
    for i in range(ncls1):
        for j in range(ncls2):
            _f1 = f1[i]
            _f2 = f2[j]
            _a1, _b1 = _f1
            _a2, _b2 = _f2
            if _a2 == "G_B" and _b2 == "G_E":
                _a2, _b2 = "G_E", "G_B"
                i2, j2 = j2, i2
            _covkey = _a1, _b1, _a2, _b2, i1, j1, i2, j2
            covkeys[(i, j)] = _covkey
    return covkeys, f1, f2


def cov2spinblocks(cls, cov):
    _covs = {}
    cls_keys = list(cls.keys())
    for i in range(0, len(cls_keys)):
        for j in range(i, len(cls_keys)):
            k1 = cls_keys[i]
            k2 = cls_keys[j]
            ell1 = cls[k1].ell
            ell2 = cls[k2].ell
            cl1 = np.atleast_2d(cls[k1])
            cl2 = np.atleast_2d(cls[k2])
            ncls1, nells1 = cl1.shape
            ncls2, nells2 = cl2.shape
            A, B, nA, nB = k1[0], k1[1], k1[2], k1[3]
            C, D, nC, nD = k2[0], k2[1], k2[2], k2[3]

            covkey = (A, B, C, D, nA, nB, nC, nD)
            # Writes the covkeys of the spin components associated with covkey
            # it also returns what fields go in what axis
            #  comps1     (E, E) (B, B) (E,B) <--- comps2
            # (POS, E)
            # (POS, B)
            covkeys, comps1, comps2 = split_comps(covkey, ncls1, ncls2)
            # Save comps in dtype metadata
            dt = np.dtype(
                float,
                metadata={
                    "fields1": comps1,
                    "fields2": comps2,
                },
            )
            _cov = np.zeros((ncls1, ncls2, nells1, nells2), dtype=dt)
            for i in range(ncls1):
                for j in range(ncls2):
                    _covkey = covkeys[(i, j)]
                    if _covkey not in cov.keys():
                        # This triggers if the element doesn't exist
                        # but the symmetrical term does
                        _cov[i, j, :, :] = np.zeros((nells2, nells1))
                    else:
                        _cov[i, j, :, :] = cov[_covkey]
            _covs[covkey] = Result(_cov, ell=(ell1, ell2))
    return _covs
