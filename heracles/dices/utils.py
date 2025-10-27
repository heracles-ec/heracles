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


def get_cl(key, cls):
    """
    Internal method to get a Cl from a dictionary of Cls.
    Check if the key exists if not tries to find the symmetric key.
    input:
        key: key of the Cl
        cls: dictionary of Cls
    returns:
        cl: Cl
    """
    if key in cls:
        return cls[key].array
    else:
        a, b, i, j = key
        key_sym = (b, a, j, i)
        if key_sym in cls:
            arr = cls[key_sym].array
            s1, s2 = cls[key_sym].spin
            if s1 != 0 and s2 != 0:
                print("dims of arr:", key_sym, arr.shape)
                return np.transpose(arr, axes=(1, 0, 2))
            else:
                return arr

        else:
            raise KeyError(f"Key {key} not found in Cls.")


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
        _Cls[key] = replace(Cls[key], array=arr)
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
        _Cls[key] = replace(Cls[key], array=arr)
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
        cov_c[key] = replace(a, array=c)
    return cov_c
