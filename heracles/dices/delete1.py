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
from .utils_cl import (
    get_Clkey,
    mat2dict,
    dict2mat,
    compsep_Cls,
    get_Cl_mu,
    cov2corr,
)
from .bias_corrrection import (
    get_bias,
    add_to_Cls,
)


def get_delete1_cov(Cls0, Clsjks, shrink=True):
    """
    Internal method to compute the shrunk covariance.
    inputs:
        Cls0 (dict): Dictionary of data Cls
        Clsjks (dict): Dictionary of delete1 data Cls
        Clsjks_wbias (dict): Dictionary of delete1 data Cls
    returns:
        shrunk_cov (dict): Dictionary of shrunk delete1 covariance
        delete1_cov (dict): Dictionary of delete1 covariance
        target_cov (dict): Dictionary of target covariance
    """
    # Get JackNJk
    JackNjk = len(Clsjks.keys())

    # Add bias to Cls
    Clsjks_wbias = {}
    for key in list(Clsjks.keys()):
        Cljk = Clsjks[key]
        biasjk = get_bias(Cljk)
        Clsjks_wbias[key] = add_to_Cls(Cljk, biasjk)

    # Separate component Cls
    Cqs0 = compsep_Cls(Cls0)
    Cqsjks = {}
    Cqsjks_wbias = {}
    for key in list(Clsjks.keys()):
        Clsjk = Clsjks[key]
        Clsjk_wbias = Clsjks_wbias[key]
        Cqsjks[key] = compsep_Cls(Clsjk)
        Cqsjks_wbias[key] = compsep_Cls(Clsjk_wbias)

    # Mean Cls
    Cqsjks_mu = get_Cl_mu(Cqsjks)
    Cqsjks_mu_wbias = get_Cl_mu(Cqsjks_wbias)

    # Concatenate Cls
    Cqsjks_all = []
    for key in Cqsjks.keys():
        cls = Cqsjks[key]
        cls_all = np.concatenate([cls[key] for key in list(cls.keys())])
        Cqsjks_all.append(cls_all)
    Cqsjks_mu_all = np.mean(np.array(Cqsjks_all), axis=0)

    # W matrices
    W = get_W(Cqsjks_all, Cqsjks_mu_all, jk=True)
    Wbar = np.mean(W, axis=0)

    # Compute Jackknife covariance
    S = (JackNjk / (JackNjk - 1)) * Wbar

    # Compute target matrix
    ClGauss_cov = get_gaussian_cov(Cqsjks_mu_wbias)
    ClGauss_fullcov = dict2mat(Cqs0, ClGauss_cov)
    ClGauss_corr = cov2corr(ClGauss_fullcov)
    T = get_target_cov(S, ClGauss_corr)

    # Compute scalar shrinkage intensity
    if shrink:
        lambda_star = get_lambda_star_single_rbar(S, W, Wbar, ClGauss_corr)
    else:
        print("Shrinkage intensity not implemented for unbinned Cls")
        lambda_star = 0.0
    print("Shrinkage intensity = %0.4f" % lambda_star)

    # Apply shrinkage
    shrunk_S = lambda_star * T + (1 - lambda_star) * S

    # To dictionaries
    shrunk_S = mat2dict(Cqs0, shrunk_S)
    T = mat2dict(Cqs0, T)
    S = mat2dict(Cqs0, S)
    return shrunk_S, S, T


def get_W(x, xbar, jk=False):
    """
    Computes the W matrices used to cpmpute the covariance
    of the covariance matrix estimate.
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


def get_target_cov(S, rbar):
    """
    Computes the estimate of the target matrix.
    input:
        S: target covariance matrix
        rbar: Gaussian correlation matrix
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


def get_gaussian_cov(Cl_mu):
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
        S: target covariance matrix
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


def get_lambda_star_single_rbar(S, W, Wbar, rbar):
    """
    Computes the optimal linear shrinkage factor.
    input:
        S: target covariance matrix
        W: W matrices
        Wbar: mean W matrix
        rbar: correlation matrix
    returns:
        lambda_star: optimal linear shrinkage factor
    """
    f = get_f(S, W, Wbar)
    numerator = 0.0
    denominator = 0.0
    for i in range(0, len(S)):
        for j in range(0, len(S)):
            if i != j:
                numerator += get_covSS(i, j, i, j, W, Wbar) - rbar[i, j] * f[i, j]
                denominator += (
                    S[i, j] - rbar[i, j] * np.sqrt(S[i, i] * S[j, j])
                ) ** 2.0
    lambda_star = numerator / denominator
    return lambda_star
