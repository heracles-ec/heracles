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
from ..result import Result
from .delete1 import jackknife_covariance


def delete2_correction(Cls0, Cls1, Cls2):
    """
    Internal method to compute the delete2 correction.
    inputs:
        Cls0 (dict): Dictionary of data Cls
        Cls1 (dict): Dictionary of delete1 data Cls
        Cls2 (dict): Dictionary of delete2 data Cls
    returns:
        Q (dict): Dictionary of delete2 correction
    """
    Q_ii = []
    Njk = len(Cls1)
    for kk in Cls2.keys():
        k1, k2 = kk
        qii = {}
        for key in Cls2[kk].keys():
            _qii = Njk * Cls0[key].array
            _qii -= (Njk - 1) * Cls1[(k1,)][key].array 
            _qii -= (Njk - 1) * Cls1[(k2,)][key].array
            _qii += (Njk - 2) * Cls2[kk][key].array
            _qii = Result(_qii, ell=Cls0[key].ell)
            qii[key] = _qii
            Q_ii.append(qii)
    Q = jackknife_covariance(Q_ii, nd=2)
    return Q


def debias_covariance(cov_jk, Cls0, Clsjks, Clsjk2s):
    """
    Debiases the Jackknife covariance using the delete2 ensemble.
    inputs:
        cov_jk (dict): Dictionary of delete1 covariance
        Cls0 (dict): Dictionary of data Cls
        Clsjks (dict): Dictionary of delete1 data Cls
        Clsjk2s (dict): Dictionary of delete2 data Cls
    returns:
        debiased_cov (dict): Dictionary of debiased Jackknife covariance
    """
    Q = delete2_correction(Cls0, Clsjks, Clsjk2s)
    debiased_cov = {}
    for key in list(cov_jk.keys()):
        c = cov_jk[key].array - Q[key].array
        debiased_cov[key] = Result(
            c,
            ell=cov_jk[key].ell,
            axis=cov_jk[key].axis,
        )
    return debiased_cov
