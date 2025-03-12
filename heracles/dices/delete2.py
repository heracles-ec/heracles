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
    Fields2Components,
    get_Cl_cov,
)


def get_delete2_correction(Cls0, Clsjks, Clsjk2s):
    """
    Internal method to compute the delete2 covariance.
    inputs:
        Cls0 (dict): Dictionary of data Cls
        Clsjks (dict): Dictionary of delete1 data Cls
        Clsjk2s (dict): Dictionary of delete2 data Cls
    returns:
        Cljk_cov (dict): Dictionary of delete2 covariance
    """
    # Get JackNJk
    JackNjk = len(Clsjks.keys())

    # Bin Cls
    Cqs0 = Fields2Components(Cls0)
    Cqsjks = []
    for key in Clsjks.keys():
        Cqsjks.append(Fields2Components(Clsjks[key]))

    jk1 = []
    jk2 = []
    Cqsjks2 = []
    for jk in range(1, JackNjk):
        _jk2 = np.arange(jk + 1, JackNjk + 1)
        _jk1 = jk * np.ones(len(_jk2))
        _jk1 = _jk1.astype("int")
        _jk2 = _jk2.astype("int")
        _Clsjks = []
        for __jk2 in _jk2:
            cqs = Clsjk2s[(jk, __jk2)]
            _Clsjks.append(Fields2Components(cqs))
        jk1.append(_jk1)
        jk2.append(_jk2)
        [Cqsjks2.append(_Cls) for _Cls in _Clsjks]
    jk1 = np.concatenate(jk1)
    jk2 = np.concatenate(jk2)

    # Compute bias correction
    Qii = []
    for i in range(0, len(Cqsjks2)):
        i1 = jk1[i]
        i2 = jk2[i]
        _Qii = {}
        for key in list(Cqs0.keys()):
            __Qii = JackNjk * Cqs0[key].__array__()
            __Qii -= (JackNjk - 1) * (
                Cqsjks[i1 - 1][key].__array__() + Cqsjks[i2 - 1][key].__array__()
            )
            __Qii += (JackNjk - 2) * Cqsjks2[i][key].__array__()
            _Qii[key] = __Qii
        Qii.append(_Qii)

    n = JackNjk * (JackNjk - 1) / 2
    Q_cov = get_Cl_cov(Qii)
    for key in Q_cov.keys():
        Q_cov[key] *= n - 1
        Q_cov[key] *= 1 / (JackNjk * (JackNjk + 1))
    return Q_cov


def get_delete2_cov(delete1_cov, Cls0, Clsjks, Clsjk2s):
    """
    Internal method to compute the delete2 covariance.
    inputs:
        delete1_cov (dict): Dictionary of delete1 covariance
        Cls0 (dict): Dictionary of data Cls
        Clsjks (dict): Dictionary of delete1 data Cls
        Clsjk2s (dict): Dictionary of delete2 data Cls
    returns:
        delete2_cov (dict): Dictionary of delete2 covariance
    """
    Q = get_delete2_correction(Cls0, Clsjks, Clsjk2s)
    delete2_cov = {}
    for key in list(delete1_cov.keys()):
        delete2_cov[key] = delete1_cov[key] - Q[key]
    return delete2_cov
