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
#####
# WIP
#####
import copy
import itertools
import numpy as np
from ..result import Result


def flatten(results, order=None):
    """
    Flattens the results dictionary into a concatenated array.
    input:
        results: dictionary of Cl values
        order: list of keys to use for the flattening
    """
    components_dict = _fields2components(results)
    data = _components2data(components_dict, order=order)
    return data


def _fields2components(results):
    """
    Separates the SHE values in *results* into E and B modes.
    """
    # It feels like all of this should be able to be done programmatically
    # but I don't know how to do that yet.
    _results = {}
    for key in list(results.keys()):
        r = copy.copy(results[key])
        ell = r.ell
        axis = r.axis
        if len(axis) == 1:
            # We are dealing with Cls
            a1, b1, i1, j1 = key
            sa1, sb1 = r.spin
            _a1, idx1 = _split_key(a1, sa1)
            _b1, idx2 = _split_key(b1, sb1)
            for k, idx in zip(
                itertools.product(_a1, _b1), itertools.product(idx1, idx2)
            ):
                __a1, __b1 = k
                _key = (__a1, __b1, i1, j1)
                _r = r[idx]
                _r = np.squeeze(_r)
                if sa1 == 2 and sb1 == 2:
                    mode1, mode2 = __a1[-1], __b1[-1]
                    duplicate_cond = mode1 == "B" and mode2 == "E" and i1 == j1
                else:
                    duplicate_cond = False
                if not duplicate_cond:
                    _results[_key] = Result(_r, spin=(sa1, sb1), axis=(0,), ell=ell)
        elif len(axis) == 2:
            # We are dealing with Covariance matrices
            a1, b1, a2, b2, i1, j1, i2, j2 = key
            sa1, sb1, sa2, sb2 = r.spin
            _a1, idx1 = _split_key(a1, sa1)
            _b1, idx2 = _split_key(b1, sb1)
            _a2, idx3 = _split_key(a2, sa2)
            _b2, idx4 = _split_key(b2, sb2)
            for k, idx in zip(
                itertools.product(_a1, _b1, _a2, _b2),
                itertools.product(idx1, idx2, idx3, idx4),
            ):
                __a1, __b1, __a2, __b2 = k
                _key = (__a1, __b1, __a2, __b2, i1, j1, i2, j2)
                _r = r[idx]
                _r = np.squeeze(_r)
                _results[_key] = Result(
                    _r, spin=(sa1, sb1, sa2, sb2), axis=(0, 1), ell=ell
                )
        else:
            raise ValueError(
                "Results with more than 3 axes are not supported at the moment."
            )
    return _results


def _components2data(results, order=None):
    naxis = np.unique([len(result.axis) for result in results.values()])
    if len(naxis) != 1:
        raise ValueError("Different types of results in the same dictionary.")
    naxis = naxis[0]
    if naxis == 1:
        # We are dealing with Cls
        if order is None:
            order = list(results.keys())
        ells = [results[key].ell for key in order]
        nells = [len(ell) for ell in ells]
        nells = np.sum(nells)
        data = np.zeros((nells))
        for i, key in enumerate(order):
            ell = results[key].ell
            nells = len(ell)
            data[i * nells : (i + 1) * nells] = results[key]
    if naxis == 2:
        # We are dealing with Covariance matrices
        if order is None:
            order = []
            nells = []
            for key in list(results.keys()):
                s1, s2, _, _ = results[key].spin
                ell = results[key].ell
                nell = len(ell[0])
                _key = (key[0], key[1], key[4], key[5])
                a, b, i, j = _key
                if s1 == 2 and s2 == 2:
                    mode1, mode2 = _key[-1], _key[-1]
                    duplicate_cond = mode1 == "B" and mode2 == "E" and i == j
                else:
                    duplicate_cond = False
                if _key not in order and not duplicate_cond:
                    order.append(_key)
                    nells.append(nell)
        else:
            nells = []
            for key in order:
                covkey = (
                    key[0],
                    key[1],
                    key[0],
                    key[1],
                    key[2],
                    key[3],
                    key[2],
                    key[3],
                )
                ell = results[covkey].ell
                nell = len(ell[0])
                nells.append(nell)
        data = np.zeros((np.sum(nells), np.sum(nells)))
        for i, ki in enumerate(order):
            for j, kj in enumerate(order):
                if i <= j:
                    # Fill in lower triangle
                    a1, b1, i1, j1 = ki
                    a2, b2, i2, j2 = kj
                    covkey = (a1, b1, a2, b2, i1, j1, i2, j2)
                    if covkey not in results.keys():
                        # Happens if the BBAA exists but not AABB
                        covkey = (a2, b2, a1, b1, i2, j2, i1, j1)
                    c = results[covkey].array
                    size_i = nells[i]
                    size_j = nells[j]
                    data[
                        i * size_i : (i + 1) * size_i, j * size_j : (j + 1) * size_j
                    ] = c
                    if i != j:
                        data[
                            j * size_j : (j + 1) * size_j, i * size_i : (i + 1) * size_i
                        ] = c.T
        # Fill in upper triangle
        data = np.tril(data) + np.tril(data, -1).T
    return data


def _split_key(f, spin, pos=None):
    if spin == 0:
        return [f], [pos]
    if spin == 2:
        return [f + "_E", f + "_B"], [0, 1]
