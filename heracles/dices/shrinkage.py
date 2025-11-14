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
        sa1, sb1 = cl1.spin
        sa2, sb2 = cl2.spin
        # dof spins
        dof_a1 = 1 if sa1 == 0 else 2
        dof_b1 = 1 if sb1 == 0 else 2
        dof_a2 = 1 if sa2 == 0 else 2
        dof_b2 = 1 if sb2 == 0 else 2
        # get attributes of result
        ell1 = get_result_array(cl1, "ell")[0]
        ell2 = get_result_array(cl2, "ell")[0]
        # keys for cov
        _key1 = (a1, a2, i1, i2)
        _key2 = (b1, b2, j1, j2)
        _key3 = (a1, b2, i1, j2)
        _key4 = (b1, a2, j1, i2)
        _cl1 = get_cl(_key1, cls)
        _cl2 = get_cl(_key2, cls)
        _cl3 = get_cl(_key3, cls)
        _cl4 = get_cl(_key4, cls)
        # add dimension if needed
        _cl1 = _cl1 if _cl1.ndim > 1 else _cl1[None, :]
        _cl2 = _cl2 if _cl2.ndim > 1 else _cl2[None, :]
        _cl3 = _cl3 if _cl3.ndim > 1 else _cl3[None, :]
        _cl4 = _cl4 if _cl4.ndim > 1 else _cl4[None, :]
        # add dimension if needed 
        _cl1 = _cl1 if _cl1.ndim > 2 else _cl1[None, :]
        _cl2 = _cl2 if _cl2.ndim > 2 else _cl2[None, :]
        _cl3 = _cl3 if _cl3.ndim > 2 else _cl3[None, :]
        _cl4 = _cl4 if _cl4.ndim > 2 else _cl4[None, :]
        idx1 = np.arange(dof_a1)
        idx2 = np.arange(dof_b1)
        idx3 = np.arange(dof_a2)
        idx4 = np.arange(dof_b2)
        combos = list(itertools.product(*[idx1, idx2, idx3, idx4]))
        # shape of the result
        r = np.zeros((dof_a1, dof_b1, dof_a2, dof_b2, len(ell1)))
        for combo in combos:
            _idx1, _idx2, _idx3, _idx4 = combo
            try:
                __cl1 = _cl1[_idx1, _idx3, :]
            except Exception:
                __cl1 = _cl1[_idx3, _idx1, :]
            try:
                __cl2 = _cl2[_idx2, _idx4, :]
            except Exception:
                __cl2 = _cl2[_idx4, _idx2, :]
            try:
                __cl3 = _cl3[_idx1, _idx4, :]
            except Exception:
                __cl3 = _cl3[_idx4, _idx1, :]
            try:
                __cl4 = _cl4[_idx2, _idx3, :]
            except Exception:
                __cl4 = _cl4[_idx3, _idx2, :]
            _cov = __cl1 * __cl2 + __cl3 * __cl4
            r[_idx1, _idx2, _idx3, _idx4, :] = _cov
        r = np.squeeze(r)
        eye = np.eye(r.shape[-1])
        r = r[..., :, None] * eye
        # Assign to cov
        _ax = np.arange(len(r.shape))
        ax1, ax2 = int(_ax[-2]), int(_ax[-1])
        cov[covkey] = Result(
            r, spin=(sa1, sb1, sa2, sb2), ell=(ell1, ell2), axis=(ax1, ax2)
        )
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
