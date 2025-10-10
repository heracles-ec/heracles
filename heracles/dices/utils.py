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
from ..result import _update_result_array


def add_to_Cls(Cls, x):
    """
    Adds a dictionary of Cl values to another.
    input:
        Cls: dictionary of Cl values
        x: dictionary of Cl values
    returns:
        Cls: updated dictionary of Cl values
    """
    _Cls = {}
    for key in Cls.keys():
        arr = Cls[key].array + x[key]
        _Cls[key] = _update_result_array(Cls[key], arr)
    return _Cls


def sub_to_Cls(Cls, x):
    """
    Substracts a dictionary of Cl values to another.
    input:
        Cls: dictionary of Cl values
        x: dictionary of Cl values
    returns:
        Cls: updated dictionary of Cl values
    """
    _Cls = {}
    for key in Cls.keys():
        arr = Cls[key].array - x[key]
        _Cls[key] = _update_result_array(Cls[key], arr)
    return _Cls


def impose_correlation(cov_a, cov_b):
    """
    Imposes the correlation of b to a.
    input:
        a: dictionary of covariance matrices
        b: dictionary of covariance matrices
    returns:
        a: updated dictionary of covariance matrices
    """
    cov_c = {}
    for key in cov_a:
        a = cov_a[key]
        b = cov_b[key]
        a_v = np.diagonal(a, axis1=-2, axis2=-1)
        b_v = np.diagonal(b, axis1=-2, axis2=-1)
        a_std = np.sqrt(a_v[..., None, :])
        b_std = np.sqrt(b_v[..., None, :])
        c = a * (b_std * np.swapaxes(b_std, -1, -2))
        c /= a_std * np.swapaxes(a_std, -1, -2)
        cov_c[key] = _update_result_array(a, c)
    return cov_c
