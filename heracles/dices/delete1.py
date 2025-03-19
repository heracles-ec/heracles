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
from .utils import (
    get_Clkey,
    get_Cl_mu,
    get_Wbar,
    get_W,
    cov2corr,
)
from .bias_correction import (
    get_bias,
    add_to_Cls,
)
from .io import (
    Fields2Components,
    Data2Components,
    Components2Data,
)


def get_jackknife_cov(Cls0, Clsjks):
    """
    Computes the jackknife covariance matrix.
    inputs:
        Cls0 (dict): Dictionary of data Cls
        Clsjks (dict): Dictionary of delete1 data Cls
    returns:
        cov_jk (dict): Dictionary of delete1 covariance
    """
    # Get JackNJk
    JackNjk = len(Clsjks.keys())
    # Component separate
    Cqs0 = Fields2Components(Cls0)
    Cqsjks = {}
    for key in list(Clsjks.keys()):
        Clsjk = Clsjks[key]
        Cqsjks[key] = Fields2Components(Clsjk)
    # Concatenate Cls
    Cqsjks_all = []
    for key in Cqsjks.keys():
        cls = Cqsjks[key]
        cls_all = np.concatenate([cls[key] for key in list(cls.keys())])
        Cqsjks_all.append(cls_all)
    Cqsjks_mu_all = np.mean(np.array(Cqsjks_all), axis=0)
    # W matrices
    Wbar = get_Wbar(Cqsjks_all, Cqsjks_mu_all)
    # Compute Jackknife covariance
    cov_jk = (JackNjk - 1) * Wbar
    # Data vector to dictionary
    cov_jk = Data2Components(Cqs0, cov_jk)
    return cov_jk


def shrink_cov(Cls0, cov, target, shrinkage_factor):
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


def get_gaussian_cov(Clsjks):
    """
    Computes Gaussian covariance from delete1 Cls.
    inputs:
        Clsjks (dict): Dictionary of delete1 data Cls
    returns:
        target_cov (dict): Dictionary of target covariance
    """
    # Add bias to Cls
    Clsjks_wbias = {}
    for key in list(Clsjks.keys()):
        Cljk = Clsjks[key]
        biasjk = get_bias(Cljk)
        Clsjks_wbias[key] = add_to_Cls(Cljk, biasjk)

    # Separate component Cls
    Cqsjks_wbias = {}
    for key in list(Clsjks_wbias.keys()):
        Clsjk_wbias = Clsjks_wbias[key]
        Cqsjks_wbias[key] = Fields2Components(Clsjk_wbias)

    # Compute target matrix
    Cqsjks_mu_wbias = get_Cl_mu(Cqsjks_wbias)
    target = _get_gaussian_cov(Cqsjks_mu_wbias)
    return target


def _get_gaussian_cov(Cl_mu):
    """
    Computes Gaussian estimate of the target matrix.
    input:
        Cl_mu: power spectrum
        covkeys: list of covariance keys
    returns:
        T: target matrix
    """
    T = {}
    Cl_keys = list(Cl_mu.keys())
    for i in range(0, len(Cl_keys)):
        for j in range(i, len(Cl_keys)):
            ki = Cl_keys[i]
            kj = Cl_keys[j]
            A, B, nA, nB = ki[0], ki[1], ki[2], ki[3]
            C, D, nC, nD = kj[0], kj[1], kj[2], kj[3]
            covkey = (A, B, C, D, nA, nB, nC, nD)
            k = [A, nA]
            q = [B, nB]
            m = [C, nC]
            n = [D, nD]

            clkey1 = get_Clkey(k, m)
            clkey2 = get_Clkey(q, n)
            clkey3 = get_Clkey(k, n)
            clkey4 = get_Clkey(q, m)

            cl1 = Cl_mu[clkey1].__array__()
            cl2 = Cl_mu[clkey2].__array__()
            cl3 = Cl_mu[clkey3].__array__()
            cl4 = Cl_mu[clkey4].__array__()

            Cl_diag = cl1 * cl2 + cl3 * cl4
            # (2*ls+1) term not needed since we only
            # care about the correlation matrix
            # Cl_diag /= (2*ls + 1)*fsky*dl
            _T = np.zeros((len(Cl_diag), len(Cl_diag)))
            ind = np.arange(len(Cl_diag))
            _T[ind, ind] = Cl_diag
            T[covkey] = _T
    return T


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


def get_shrinkage_factor(cls0, Clsjks, target):
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
