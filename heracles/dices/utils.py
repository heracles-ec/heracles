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

try:
    from copy import replace
except ImportError:
    # Python < 3.13
    from dataclasses import replace


def add_to_Cls(cls, x):
    """
    Adds a dictionary of Cl values to another.
    input:
        Cls: dictionary of Cl values
        x: dictionary of Cl values
    returns:
        Cls: updated dictionary of Cl values
    """
    _cls = {}
    for key in cls.keys():
        arr = cls[key].array + x[key]
        _cls[key] = replace(cls[key], array=arr)
    return _cls


def sub_to_Cls(cls, x):
    """
    Substracts a dictionary of Cl values to another.
    input:
        Cls: dictionary of Cl values
        x: dictionary of Cl values
    returns:
        Cls: updated dictionary of Cl values
    """
    _cls = {}
    for key in cls.keys():
        arr = cls[key].array - x[key]
        _cls[key] = replace(cls[key], array=arr)
    return _cls


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
        cov_c[key] = replace(cov_a[key], array=c)
    return cov_c
