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
from .utils import (
    expand_spin0_dims,
    squeeze_spin0_dims,
)
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
    get_cl,
    flatten,
)

try:
    from copy import replace
except ImportError:
    # Python < 3.13
    from dataclasses import replace


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
        shrunk_cov[key] = replace(cov[key], array=sc)
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
    cls1_all = [flatten(cls1[key]) for key in list(cls1.keys())]
    cls1_mu_all = np.mean(np.array(cls1_all), axis=0)
    target = flatten(target)
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


def gaussian_covariance(cls):
    b = bias(cls)
    cls = add_to_Cls(cls, b)
    cov = {}
    for key1, key2 in itertools.combinations_with_replacement(cls.keys(), 2):
        a1, b1, i1, j1 = key1
        a2, b2, i2, j2 = key2
        covkey = (a1, b1, a2, b2, i1, j1, i2, j2)
        # get reference results
        cl1 = cls[key1]
        cl2 = cls[key2]
        # get attributes of result
        (ell1,) = get_result_array(cl1, "ell")
        (ell2,) = get_result_array(cl2, "ell")
        # get dof by expanding spin0 dims
        # Eg. POS POS goes from (l) to (1, 1, l)
        cl1 = expand_spin0_dims(cls[key1])
        cl2 = expand_spin0_dims(cls[key2])
        # get cls needed
        _cl1 = get_cl((a1, a2, i1, i2), cls)
        _cl2 = get_cl((b1, b2, j1, j2), cls)
        _cl3 = get_cl((a1, b2, i1, j2), cls)
        _cl4 = get_cl((b1, a2, j1, i2), cls)
        # get spins
        _cl1 = expand_spin0_dims(_cl1)
        _cl2 = expand_spin0_dims(_cl2)
        _cl3 = expand_spin0_dims(_cl3)
        _cl4 = expand_spin0_dims(_cl4)
        # shape of the result
        dof_a1, dof_b1, _ = cl1.shape
        dof_a2, dof_b2, _ = cl2.shape
        _ell = min(len(ell1), len(ell2))
        r = np.zeros((dof_a1, dof_b1, dof_a2, dof_b2, _ell))
        for _1, _2, _3, _4 in np.ndindex(r.shape[:-1]):
            _r = _cl1[_1, _3] * _cl2[_2, _4] + _cl3[_1, _4] * _cl4[_2, _3]
            r[_1, _2, _3, _4, :] = _r
        # make diagonal into matrix
        eye = np.eye(r.shape[-1])
        r = r[..., :, None] * eye
        # Assign to cov
        r = Result(r, spin=(*cl1.spin, *cl2.spin), ell=(ell1, ell2), axis=(-2, -1))
        # squeeze spin0 dims
        cov[covkey] = squeeze_spin0_dims(r)
    return cov


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
