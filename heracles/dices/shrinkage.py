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
import itertools
from ..result import (
    Result,
    get_result_array,
)
from .jackknife import (
    bias,
)
from .utils import (
    add_to_Cls,
)
from .io import (
    Fields2Components,
    Components2Data,
    format_key,
    _split_comps,
)


def shrink_covariance(cov, target, shrinkage_factor):
    """
    Internal method to compute the shrunk covariance.
    inputs:
        cov (dict): Dictionary of Jackknife covariance
        target (dict): Dictionary of target covariance
        shrinkage_factor (float): Shrinkage factor
    returns:
        shrunk_cov (dict): Dictionary of shrunk delete1 covariance
    """
    shrunk_cov = {}
    for key in list(cov.keys()):
        c = cov[key]
        t = target[key]
        n, m, nell1, nell1 = np.shape(c)
        # Correlate target covariance
        tc = np.zeros((n, m, nell1, nell1))
        for i in range(0, n):
            for j in range(0, m):
                c_ij = c[i, j, :, :]
                t_ij = t[i, j, :, :]
                tc_ij = correlate_target(c_ij, t_ij)
                tc[i, j, :, :] = tc_ij
        # Shrink covariance
        r = shrinkage_factor * tc + (1 - shrinkage_factor) * c
        shrunk_cov[key] = Result(r, axis=(0, 1), ell=c.ell)
    return shrunk_cov


def correlate_target(cov, target):
    """
    Computes the estimate of the target matrix.
    input:
        S (array): target covariance matrix
        rbar (array): Gaussian correlation matrix
    returns:
        T: correlation matrix of S
    """
    T = np.zeros(np.shape(cov))
    for i in range(0, len(cov)):
        for j in range(0, len(cov)):
            if i == j:
                T[i, j] = cov[i, j]
            else:
                T[i, j] = target[i, j] * np.sqrt(cov[i, i] * cov[j, j])
                T[i, j] /= np.sqrt(target[i, i] * target[j, j])
    return T


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


def covW(i1, j1, i2, j2, W, Wbar):
    """
    Computes the covariance of the W matrices.
    input:
        i, j, l, m: indices
        W: W matrices
        Wbar: mean W matrix
    returns:
        covSS: covariance of W matrices
    """
    n = len(W)
    covSS = 0.0
    for k in range(0, len(W)):
        covSS += (W[k][i1, j1] - Wbar[i1, j1]) * (W[k][i2, j2] - Wbar[i2, j2])
    covSS *= n / ((n - 1) ** 3.0)
    return covSS


def shrinkage_factor(cls1, target):
    """
    Computes the optimal linear shrinkage factor.
    input:
        cls1: delete1 Cls
        target: target matrix
    returns:
        lambda_star: optimal linear shrinkage factor
    """
    # Separate component Cls
    _cls1 = {}
    for key in list(cls1.keys()):
        cl = cls1[key]
        _cls1[key] = Fields2Components(cl)
    target = Fields2Components(target)
    # To data vector
    cls1_all = [Components2Data(_cls1[key]) for key in list(cls1.keys())]
    cls1_mu_all = np.mean(np.array(cls1_all), axis=0)
    target = Components2Data(target)
    # Ingredient for the shrinkage factor
    Njk = len(cls1_all)
    W = get_W(cls1_all, cls1_mu_all)
    Wbar = np.mean(W, axis=0)
    S = (Njk - 1) * Wbar
    target_corr = target
    target_corr /= np.outer(np.sqrt(np.diag(target)), np.sqrt(np.diag(target)))
    # Compute shrinkage factor
    numerator = 0.0
    denominator = 0.0
    for i in range(0, len(S)):
        for j in range(0, len(S)):
            if i != j:
                f = 0.5 * np.sqrt(Wbar[j, j] / Wbar[i, i]) * covW(i, i, i, j, W, Wbar)
                f += 0.5 * np.sqrt(Wbar[i, i] / Wbar[j, j]) * covW(j, j, i, j, W, Wbar)
                t = target_corr[i, j]
                numerator += covW(i, j, i, j, W, Wbar) - t * f
                denominator += (S[i, j] - t * np.sqrt(S[i, i] * S[j, j])) ** 2
    lambda_star = numerator / denominator
    return lambda_star


def gaussian_covariance(Cls):
    """
    Computes Gaussian estimate of the target matrix.
    input:
        Cls: power spectra
    returns:
        T: target matrix
    """
    # Add bias to Cls
    b = bias(Cls)
    Cls = add_to_Cls(Cls, b)
    # Separate Cls into Cls
    _Cls = Fields2Components(Cls)
    # Compute Gaussian covariance
    cov = {}
    for key1, key2 in itertools.combinations_with_replacement(Cls, 2):
        # covariance key
        a1, b1, i1, j1 = key1
        a2, b2, i2, j2 = key2
        covkey = (a1, b1, a2, b2, i1, j1, i2, j2)
        # get reference results
        cl1 = Cls[key1]
        cl2 = Cls[key2]
        # get components
        comps1 = _split_comps(key1)
        comps2 = _split_comps(key2)
        # get attributes of result
        ell1 = get_result_array(cl1, "ell")
        ell2 = get_result_array(cl2, "ell")
        ell = ell1 + ell2
        r = np.zeros((len(comps1), len(comps2), len(ell1[0]), len(ell2[0])))
        # get covariance
        for i, comp1 in enumerate(comps1):
            for j, comp2 in enumerate(comps2):
                _a1, _b1, _i1, _j1 = comp1
                _a2, _b2, _i2, _j2 = comp2
                key = (_a1, _b1, _a2, _b2, _i1, _j1, _i2, _j2)
                _cov = _gaussian_covariance(_Cls, key)
                r[i, j, :, :] = np.diag(_cov)
        result = Result(r, axis=(0, 1), ell=ell)
        cov[covkey] = result
    return cov


def _gaussian_covariance(cls, key):
    """
    Retunrs a particular entry of the gaussian covariance matrix.
    input:
        cls: Cls
        key: key of the entry
    returns:
        cov: covariance matrix
    """
    a1, b1, a2, b2, i1, j1, i2, j2 = key
    clkey1 = format_key((a1, a2, i1, i2))
    clkey2 = format_key((b1, b2, j1, j2))
    clkey3 = format_key((a1, b2, i1, j2))
    clkey4 = format_key((b1, a2, j1, i2))
    cl1 = cls[clkey1].array
    cl2 = cls[clkey2].array
    cl3 = cls[clkey3].array
    cl4 = cls[clkey4].array
    # Compute the Gaussian covariance
    cov = cl1 * cl2 + cl3 * cl4
    return cov