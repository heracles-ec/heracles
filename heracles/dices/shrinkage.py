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
    Data2Components,
    Components2Data,
    Components2Fields,
    format_key,
)


def shrink_covariance(Cls0, cov, target, shrinkage_factor):
    """
    Internal method to compute the shrunk covariance.
    inputs:
        Cls0 (dict): Dictionary of data Cls
        cov (dict): Dictionary of Jackknife covariance
        target (dict): Dictionary of target covariance
        shrinkage_factor (float): Shrinkage factor
    returns:
        shrunk_cov (dict): Dictionary of shrunk delete1 covariance
    """
    # Separate component Cls
    Cqs0 = Fields2Components(Cls0)
    # to matrices
    cov = Components2Data(Cqs0, cov)
    target = Components2Data(Cqs0, target)
    # Compute scalar shrinkage intensity
    target_corr = cov2corr(target)
    _target = correlate_target(cov, target_corr)
    # Apply shrinkage
    shrunk_S = shrinkage_factor * _target + (1 - shrinkage_factor) * cov
    # To dictionaries
    shrunk_S = Data2Components(Cqs0, shrunk_S)

    return shrunk_S


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


def correlate_target(S, rbar):
    """
    Computes the estimate of the target matrix.
    input:
        S (array): target covariance matrix
        rbar (array): Gaussian correlation matrix
    returns:
        T: correlation matrix of S
    """
    T = np.zeros(np.shape(S))
    for i in range(0, len(T)):
        for j in range(0, len(T)):
            if i == j:
                T[i, j] = S[i, j]
            else:
                T[i, j] = rbar[i, j] * np.sqrt(S[i, i] * S[j, j])
    return T


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
    Cls = Fields2Components(Cls)
    # Compute Gaussian covariance
    cov = {}
    for key1, key2 in itertools.combinations_with_replacement(Cls, 2):
        # get reference results
        result1 = Cls[key1]
        result2 = Cls[key2]
        # get attributes of result
        ell = get_result_array(result1, "ell")
        ell += get_result_array(result2, "ell")
        # get covariance
        a1, b1, i1, j1 = key1
        a2, b2, i2, j2 = key2
        covkey = (a1, b1, a2, b2, i1, j1, i2, j2)
        clkey1 = format_key((a1, a2, i1, i2))
        clkey2 = format_key((b1, b2, j1, j2))
        clkey3 = format_key((a1, b2, i1, j2))
        clkey4 = format_key((b1, a2, j1, i2))
        cl1 = Cls[clkey1]
        cl2 = Cls[clkey2]
        cl3 = Cls[clkey3]
        cl4 = Cls[clkey4]
        # Compute the Gaussian covariance
        _cov = cl1.array * cl2.array + cl3.array * cl4.array
        _cov = np.diag(_cov)
        # move ell axes last, in order
        ndim1 = result1.ndim
        oldaxis = result1.axis + tuple(ndim1 + ax for ax in result2.axis)
        axis = tuple(range(-len(oldaxis), 0))
        _cov = np.moveaxis(_cov, oldaxis, axis)
        result = Result(_cov, axis=axis, ell=ell)
        cov[covkey] = result
    # Turn covariance back to fields
    # cov = Components2Fields(cov)
    return cov


def get_covSS(i, j, q, m, W, Wbar):
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
        covSS += (W[k][i, j] - Wbar[i, j]) * (W[k][q, m] - Wbar[q, m])
    covSS *= n / ((n - 1) ** 3.0)
    return covSS


def get_f(S, W, Wbar):
    """
    Computes the covariance of the W matrices with the target matrix.
    input:
        S: Jackknife covariance matrix
        W: W matrices
        Wbar: mean W matrix
    returns:
        f: covariance of W matrices
    """
    f = np.zeros(np.shape(S))
    for i in range(0, len(S)):
        for j in range(0, len(S)):
            f[i, j] += np.sqrt(S[j, j] / S[i, i]) * get_covSS(i, i, i, j, W, Wbar)
            f[i, j] += np.sqrt(S[i, i] / S[j, j]) * get_covSS(j, j, i, j, W, Wbar)
            f[i, j] *= 0.5
    return f


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


def shrinkage_factor(cls0, Clsjks, target):
    """
    Computes the optimal linear shrinkage factor.
    input:
        cls0: data Cls
        Clsjks: delete1 data Cls
        target: target matrix
    returns:
        lambda_star: optimal linear shrinkage factor
    """
    # Separate component Cls
    cqs0 = Fields2Components(cls0)
    Cqsjks = {}
    for key in list(Clsjks.keys()):
        Clsjk = Clsjks[key]
        Cqsjks[key] = Fields2Components(Clsjk)
    # to matrices
    target = Components2Data(cqs0, target)
    # Compute correlation of target
    target_corr = cov2corr(target)
    # Concatenate Cls
    Cqsjks_all = []
    for key in Cqsjks.keys():
        cls = Cqsjks[key]
        cls_all = np.concatenate([cls[key] for key in list(cls.keys())])
        Cqsjks_all.append(cls_all)
    Cqsjks_mu_all = np.mean(np.array(Cqsjks_all), axis=0)

    # W matrices
    W = get_W(Cqsjks_all, Cqsjks_mu_all)
    # Compute shrinkage factor
    Njk = len(W)
    Wbar = np.mean(W, axis=0)
    S = (Njk - 1) * Wbar
    f = get_f(S, W, Wbar)
    numerator = 0.0
    denominator = 0.0
    for i in range(0, len(S)):
        for j in range(0, len(S)):
            if i != j:
                numerator += (
                    get_covSS(i, j, i, j, W, Wbar) - target_corr[i, j] * f[i, j]
                )
                denominator += (
                    S[i, j] - target_corr[i, j] * np.sqrt(S[i, i] * S[j, j])
                ) ** 2.0
    lambda_star = numerator / denominator
    return lambda_star
