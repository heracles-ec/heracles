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
)
from .io import (
    _fields2components,
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


def broadcast_multiply(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Broadcasts and multiplies two arrays A and B such that:
      - The last dimension (l) must match.
      - The output shape is A.shape[:-1] + B.shape[:-1] + (l,)

    Example:
        A.shape = (2, l)
        B.shape = (2, 2, l)
        -> result.shape = (2, 2, 2, l)
    """
    # Check compatibility
    if A.shape[-1] != B.shape[-1]:
        raise ValueError("The last dimensions of A and B must match.")

    l = A.shape[-1]

    # Expand A to match B’s prefix, and vice versa
    A_expanded = A.reshape(*A.shape[:-1], *[1] * (B.ndim - 1), l)
    B_expanded = B.reshape(*[1] * (A.ndim - 1), *B.shape[:-1], l)

    # Elementwise multiplication via broadcasting
    result = A_expanded * B_expanded

    return result


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

        # Perform the broadcasted multiplication and sum
        r = broadcast_multiply(_cl1, _cl2)
        r += broadcast_multiply(_cl3, _cl4)
        # Create an identity matrix of shape (l, l)
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
