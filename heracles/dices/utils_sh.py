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


def get_Clkey(sources1, sources2):
    """
    Produces a Cl key for data maps.
    input:
        sources1: list of sources
        sources2: list of sources
    returns:
        Clkey: Cl key
    """
    if sources1[0] == "G_E" and sources2[0] == "POS":
        Clkey = tuple([sources2[0], sources1[0], sources2[1], sources1[1]])
    elif sources1[0] == "G_B" and sources2[0] == "POS":
        Clkey = tuple([sources2[0], sources1[0], sources2[1], sources1[1]])
    elif sources1[0] == "G_B" and sources2[0] == "G_E":
        Clkey = tuple([sources2[0], sources1[0], sources2[1], sources1[1]])
    else:
        if sources1[0] == sources2[0] and sources1[1] > sources2[1]:
            Clkey = tuple([sources2[0], sources1[0], sources2[1], sources1[1]])
        else:
            Clkey = tuple([sources1[0], sources2[0], sources1[1], sources2[1]])
    return Clkey


def get_W(x, xbar, y, ybar, jk=False):
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
    _xbi, _xbj = np.meshgrid(xbar, ybar, indexing="ij")
    for i in range(0, len(x)):
        _xi, _xj = np.meshgrid(x[i], y[i], indexing="ij")
        _Wk = (_xi - _xbi) * (_xj - _xbj)
        W.append(_Wk)
    W = np.array(W)
    if jk:
        n = len(x)
        W *= ((n - 1) ** 2.0) / n
    return W


def get_T_rbar(S, rbar):
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


def get_T_new(Cl_mu, covkeys):
    """
    Computes Gaussian estimate of the target matrix.
    input:
        Cl_mu: power spectrum
        covkeys: list of covariance keys
    returns:
        T: target matrix
    """
    T = {}
    for covkey in covkeys:
        k = [covkey[0], covkey[4]]
        q = [covkey[1], covkey[5]]
        m = [covkey[2], covkey[6]]
        n = [covkey[3], covkey[7]]

        clkey1 = get_Clkey(k, m)
        clkey2 = get_Clkey(q, n)
        clkey3 = get_Clkey(k, n)
        clkey4 = get_Clkey(q, m)

        cl1 = Cl_mu[clkey1]
        cl2 = Cl_mu[clkey2]
        cl3 = Cl_mu[clkey3]
        cl4 = Cl_mu[clkey4]

        Cl_diag = cl1 * cl2 + cl3 * cl4
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
    # lambda_star = numerator / denominator
    return numerator, denominator
