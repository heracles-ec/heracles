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
    impose_correlation,
)
from .io import (
    _fields2components,
    flatten,
    _split_key,
)


def shrink(cov, target, shrinkage_factor):
    """
    Compute the shrunk covariance.
    inputs:
        cov (dict): Dictionary of Jackknife covariance
        target (dict): Dictionary of target covariance
        shrinkage_factor (float): Shrinkage factor
    returns:
        shrunk_cov (dict): Dictionary of shrunk delete1 covariance
    """
    shrunk_cov = {}
    correlated_target = impose_correlation(target, cov)
    for key in cov:
        c = cov[key].array
        tc = correlated_target[key].array
        sc = shrinkage_factor * tc + (1 - shrinkage_factor) * c
        shrunk_cov[key] = Result(sc, axis=cov[key].axis, ell=cov[key].ell)
    return shrunk_cov


def shrinkage_factor(cls1, target):
    """
    Computes the optimal linear shrinkage factor.
    input:
        cls1: delete1 Cls
        target: target matrix
    returns:
        lambda_star: optimal linear shrinkage factor
    """
    # To data vector
    cls1_first = cls1[list(cls1.keys())[0]]
    cls1_first = _fields2components(cls1_first)
    order = list(cls1_first.keys())

    cls1_all = [flatten(cls1[key]) for key in list(cls1.keys())]
    cls1_mu_all = np.mean(np.array(cls1_all), axis=0)
    target = flatten(target, order=order)
    # Ingredient for the shrinkage factor
    Njk = len(cls1_all)
    W = _get_W(cls1_all, cls1_mu_all)
    W *= (Njk - 1) ** 2 / Njk
    Wbar = np.mean(W, axis=0)
    S = (Njk / (Njk - 1)) * Wbar
    target_corr = target
    target_corr /= np.outer(np.sqrt(np.diag(target)), np.sqrt(np.diag(target)))
    # Compute shrinkage factor
    numerator = 0.0
    denominator = 0.0
    for i in range(len(S)):
        for j in range(len(S)):
            if i != j:
                f = 0.5 * np.sqrt(Wbar[j, j] / Wbar[i, i]) * _covW(i, i, i, j, W, Wbar)
                f += 0.5 * np.sqrt(Wbar[i, i] / Wbar[j, j]) * _covW(j, j, i, j, W, Wbar)
                t = target_corr[i, j]
                numerator += _covW(i, j, i, j, W, Wbar) - t * f
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
    _Cls = _fields2components(Cls)
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
        sa1, sb1 = cl1.spin
        sa2, sb2 = cl2.spin
        # get components
        _a1, idx1 = _split_key(a1, sa1, pos=0)
        _b1, idx2 = _split_key(b1, sb1, pos=0)
        _a2, idx3 = _split_key(a2, sa2, pos=0)
        _b2, idx4 = _split_key(b2, sb2, pos=0)
        # get attributes of result
        ell1 = get_result_array(cl1, "ell")
        ell2 = get_result_array(cl2, "ell")
        ell = ell1 + ell2
        r = np.zeros(
            (len(idx1), len(idx2), len(idx3), len(idx4), len(ell1[0]), len(ell2[0]))
        )
        # get covariance
        for k, idx in zip(
            itertools.product(_a1, _b1, _a2, _b2),
            itertools.product(idx1, idx2, idx3, idx4),
        ):
            __a1, __b1, __a2, __b2 = k
            _key = (__a1, __b1, __a2, __b2, i1, j1, i2, j2)
            _cov = _gaussian_covariance(_Cls, _key)
            ix1, ix2, ix3, ix4 = idx
            r[ix1, ix2, ix3, ix4, :, :] = np.diag(_cov)
        # Remove the extra dimensions
        r = np.squeeze(r)
        # Make Result
        result = Result(r, spin=(sa1, sb1, sa2, sb2), ell=ell)
        cov[covkey] = result
    return cov


def _gaussian_covariance(cls, key):
    """
    Returns a particular entry of the gaussian covariance matrix.
    input:
        cls: Cls
        key: key of the entry
    returns:
        cov: covariance matrix
    """
    a1, b1, a2, b2, i1, j1, i2, j2 = key
    cl1 = _get_cl((a1, a2, i1, i2), cls)
    cl2 = _get_cl((b1, b2, j1, j2), cls)
    cl3 = _get_cl((a1, b2, i1, j2), cls)
    cl4 = _get_cl((b1, a2, j1, i2), cls)
    cov = cl1 * cl2 + cl3 * cl4
    return cov


def _get_cl(key, cls):
    """
    Internal method to get a Cl from a dictionary of Cls.
    Check if the key exists if not tries to find the symmetric key.
    input:
        key: key of the Cl
        cls: dictionary of Cls
    returns:
        cl: Cl
    """
    if key in cls:
        return cls[key].array
    else:
        a, b, i, j = key
        key_sym = (b, a, j, i)
        if key_sym in cls:
            return cls[key_sym].array
        else:
            raise KeyError(f"Key {key} not found in Cls.")


def _get_W(x, xbar):
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
    for i in range(len(x)):
        _xi, _xj = np.meshgrid(x[i], x[i], indexing="ij")
        _Wk = (_xi - _xbi) * (_xj - _xbj)
        W.append(_Wk)
    return np.array(W)


def _covW(i1, j1, i2, j2, W, Wbar):
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
    for k in range(len(W)):
        covSS += (W[k][i1, j1] - Wbar[i1, j1]) * (W[k][i2, j2] - Wbar[i2, j2])
    covSS *= n / ((n - 1) ** 3.0)
    return covSS
