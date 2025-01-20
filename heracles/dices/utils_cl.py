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
from heracles.result import binned
from heracles.io import read
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


def get_covkeys(Cl_keys):
    """
    Produces a list of covariance keys.
    input:
        Cl_keys: list of Cl keys
    returns:
        covkeys: list of covariance keys
    """
    covkeys = []
    for i in range(0, len(Cl_keys)):
        for j in range(i, len(Cl_keys)):
            ki = Cl_keys[i]
            kj = Cl_keys[j]
            A, B, nA, nB = ki[0], ki[1], ki[2], ki[3]
            C, D, nC, nD = kj[0], kj[1], kj[2], kj[3]
            covkey = (A, B, C, D, nA, nB, nC, nD)
            covkeys.append(covkey)
    return covkeys


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
        cl = np.atleast_2d(Cls[key])
        if t1 == t2 == "POS":
            Cls_compsep[key] = cl[..., :]
        elif t1 == t2 == "SHE" and b1 == b2:
            Cls_compsep[("G_E", "G_E", b1, b2)] = cl[..., 0, :]
            Cls_compsep[("G_B", "G_B", b1, b2)] = cl[..., 1, :]
            Cls_compsep[("G_E", "G_B", b1, b2)] = cl[..., 2, :]
        elif t1 == t2 == "SHE" and b1 != b2:
            Cls_compsep[("G_E", "G_E", b1, b2)] = cl[..., 0, :]
            Cls_compsep[("G_B", "G_B", b1, b2)] = cl[..., 1, :]
            Cls_compsep[("G_E", "G_B", b1, b2)] = cl[..., 2, :]
            Cls_compsep[("G_E", "G_B", b2, b1)] = cl[..., 3, :]
        elif t1 == "POS" and t2 == "SHE":
            Cls_compsep[("POS", "G_E", b1, b2)] = cl[..., 0, :]
            Cls_compsep[("POS", "G_B", b1, b2)] = cl[..., 1, :]
    return Cls_compsep


def get_Cls_bias(Clkeys, bias_pos, bias_she):
    """
    Produces a dictionary of bias values for Cl keys.
    input:
        Clkeys: list of Cl keys
        bias_pos: list of position bias values
        bias_she: list of shear bias values
    returns:
        Cls_bias: dictionary of bias values
    """
    Cls_bias = {}
    for key in Clkeys:
        type1 = key[0]  # POS or SHE
        type2 = key[1]
        bin1 = key[2]  # 1, 2...
        bin2 = key[3]
        if type1 == type2 and bin1 == bin2:
            if type1 == "POS":
                Cls_bias[key] = bias_pos[bin1 - 1]
            elif type1 == "SHE":
                Cls_bias[key] = bias_she[bin1 - 1]
        else:
            Cls_bias[key] = 0.0
    return Cls_bias


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
        _Cls[key] = Cls[key] + x[key]
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
        _Cls[key] = Cls[key] - x[key]
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
    v[v<0.0] = np.min(np.abs(v))
    return w @ np.diag(v) @ w.T

# Summary statistics for Cls


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
