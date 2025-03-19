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
import numpy as np
from ..result import Result
from .io import Fields2Components


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


def get_W(x, xbar, jk=False):
    """
    Internal method to compute the W matrices.
    input:
        x: Cl
        xbar: mean Cl
        jk: if True, computes the jackknife version of the W matrices
    returns:
        W: W matrices
    """
    W = []
    _xbi, _xbj = np.meshgrid(xbar, xbar, indexing="ij")
    for i in range(0, len(x)):
        _xi, _xj = np.meshgrid(x[i], x[i], indexing="ij")
        _Wk = (_xi - _xbi) * (_xj - _xbj)
        W.append(_Wk)
    W = np.array(W)
    if jk:
        n = len(x)
        W *= ((n - 1) ** 2.0) / n
    return W


def get_Wbar(x, xbar):
    """
    Internal method to compute the W matrices.
    input:
        x: Cl
        xbar: mean Cl
        jk: if True, computes the jackknife version of the W matrices
    returns:
        W: W matrices
    """
    W = np.zeros((len(xbar), len(xbar)))
    n = len(x)
    _xbi, _xbj = np.meshgrid(xbar, xbar, indexing="ij")
    for i in range(0, len(x)):
        _xi, _xj = np.meshgrid(x[i], x[i], indexing="ij")
        W += (_xi - _xbi) * (_xj - _xbj)
    W *= 1/n
    return W
