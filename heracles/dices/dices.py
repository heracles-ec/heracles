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
    cov2corr,
)
from .io import (
    Fields2Components,
    Components2Data,
    Data2Components,
)


def dices_covariance(cls0, cov1, cov2):
    """
    Internal method to compute the Dices covariance.
    inputs:
        cls0 (dict): Dictionary of data Cls
        cov1 (dict): Dictionary of shrunk delete1 covariance
        cov2 (dict): Dictionary of delete2 covariance
    returns:
        dices_cov (dict): Dictionary of Dices covariance
    """
    cqs0 = Fields2Components(cls0)
    _cov1 = Components2Data(cqs0, cov1)
    _cov2 = Components2Data(cqs0, cov2)
    _corr1 = cov2corr(_cov1)
    _var1 = np.diag(_cov1).copy()
    _var2 = np.diag(_cov2).copy()
    cond = np.where(_var2 < 0)[0]
    _var2[cond] = _var1[cond]
    _sig2 = np.sqrt(_var2)
    _corr2 = np.outer(_sig2, _sig2)
    dices_cov = _corr2 * _corr1
    return Data2Components(cqs0, dices_cov)
