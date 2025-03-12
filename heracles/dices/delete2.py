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
    _get_W,
)
from .io import (
    Fields2Components,
    Data2Components,
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
    Cqs0_all = np.concatenate([Cqs0[key] for key in list(Cqs0.keys())])
    Cqsjks_all = []
    for key in Clsjks.keys():
        cls = Fields2Components(Clsjks[key])
        cls_all = np.concatenate([cls[key] for key in list(cls.keys())])
        Cqsjks_all.append(cls_all)

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
            cqs = Fields2Components(cqs)
            cqs_all = np.concatenate([cqs[key] for key in list(cqs.keys())])
            _Clsjks.append(cqs_all)
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
        _Qii = JackNjk * Cqs0_all
        _Qii -= (JackNjk - 1) * (Cqsjks_all[i1 - 1] + Cqsjks_all[i2 - 1])
        _Qii += (JackNjk - 2) * Cqsjks2[i]
        Qii.append(_Qii)

    Qii_m = np.mean(Qii, axis=0)
    Qii_W = _get_W(Qii, Qii_m)
    Q = np.mean(Qii_W, axis=0)
    n = JackNjk * (JackNjk - 1) / 2
    d = 1 / (JackNjk * (JackNjk + 1))
    Q *= (n-1)
    Q *= d
    Q = Data2Components(Cqs0, Q)
    return Q


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
