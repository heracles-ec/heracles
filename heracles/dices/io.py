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


def Fields2Components(results):
    """
    Separates the SHE values into E and B modes.
    input:
        Cls: dictionary of Cl values
    returns:
        Cls_unraveled: dictionary of Cl values
    """
    # It feels like all of this should be able to be done programmatically
    # but I don't know how to do that yet.
    _results = {}
    for key in list(results.keys()):
        r = results[key]
        ell = r.ell
        axis = r.axis
        if len(axis) == 1:
            # We are dealing with Cls
            comps = _split_comps(key)
            for i, comp in enumerate(comps):
                _r = np.atleast_2d(r.array)
                _results[comp] = Result(_r[..., i, :], ell)
        elif len(axis) == 2:
            # We are dealing with Covariance matrices
            a1, b1, a2, b2, i1, j1, i2, j2 = key
            key1 = (a1, b1, i1, j1)
            key2 = (a2, b2, i2, j2)
            comps1 = _split_comps(key1)
            comps2 = _split_comps(key2)
            for i, comp1 in enumerate(comps1):
                for j, comp2 in enumerate(comps2):
                    if i <= j:
                        # Only save the upper triangle
                        _a1, _b1, _i1, _j1 = comp1
                        _a2, _b2, _i2, _j2 = comp2
                        covkey = (_a1, _b1, _a2, _b2, _i1, _j1, _i2, _j2)
                        _r = r.array
                        _results[covkey] = Result(_r[..., i, j, :, :], ell)
        else:
            raise ValueError(
                "Results with more than 3 axes are not supported at the moment."
            )
    return _results


def Components2Data(results):
    keys = list(results.keys())
    ells = [results[key].ell for key in keys]
    if len(ells[0]) == 1:
        # We are dealing with Cls
        nells = [len(ell) for ell in ells]
        nells = np.sum(nells)
        data = np.zeros((nells))
        for i, key in enumerate(keys):
            ell = results[key].ell
            nells = len(ell)
            data[i * nells : (i + 1) * nells] = results[key]
    if len(ells[0]) == 2:
        # We are dealing with Covariance matrices
        _keys = []
        nells = []
        # Find unique keys
        for key in keys:
            ell = results[key].ell
            nell = len(results[key].ell[0])
            _key = (key[0], key[1], key[4], key[5])
            if _key not in _keys:
                _keys.append([_key])
                nells.append(nell)
        data = np.zeros((np.sum(nells), np.sum(nells)))
        for i, ki in enumerate(keys):
            for j, kj in enumerate(keys):
                if i <= j:
                    # Fill in lower triangle
                    a1, b1, i1, j1 = ki[0], ki[1], ki[2], ki[3]
                    a2, b2, i2, j2 = kj[0], kj[1], kj[2], kj[3]
                    covkey = (a1, b1, a2, b2, i1, j1, i2, j2)
                    size_i = nells[i]
                    size_j = nells[j]
                    data[
                        i * size_i : (i + 1) * size_i, j * size_j : (j + 1) * size_j
                    ] = results[covkey]
                    if i != j:
                        data[
                            j * size_j : (j + 1) * size_j, i * size_i : (i + 1) * size_i
                        ] = results[covkey].T
        # Fill in upper triangle
        data = np.tril(data) + np.tril(data, -1).T
    return data


def Data2Components(cov):
    Clkeys = list(cls.keys())
    nells = [len(cls[key].ell) for key in Clkeys]
    Cl_cov_dict = {}
    for i in range(0, len(Clkeys)):
        for j in range(i, len(Clkeys)):
            ki = Clkeys[i]
            kj = Clkeys[j]
            A, B, nA, nB = ki[0], ki[1], ki[2], ki[3]
            C, D, nC, nD = kj[0], kj[1], kj[2], kj[3]
            covkey = (A, B, C, D, nA, nB, nC, nD)
            size_i = nells[i]
            size_j = nells[j]
            Cl_cov_dict[covkey] = cov[
                i * size_i : (i + 1) * size_i, j * size_j : (j + 1) * size_j
            ]
            if i != j:
                Cl_cov_dict[covkey] = cov[
                    j * size_j : (j + 1) * size_j, i * size_i : (i + 1) * size_i
                ]
    return Cl_cov_dict


def Components2Fields(results):
    _results = {}
    keys = list(results.keys())
    naxis = np.unique([len(result.axis) for result in results.values()])
    if len(naxis) != 1:
        raise ValueError("Different types of results in the same dictionary.")
    naxis = naxis[0]
    if naxis == 1:
        # We are dealing with Cls
        # find unique fields in comps
        keys = set([_unsplit_comps(key) for key in keys])
        keys = sorted(keys)
        for key in keys:
            # get comps of each unique field
            comps = _split_comps(key)
            cls = np.array([results[format_key(comp)] for comp in comps])
            _results[key] = cls
    elif naxis == 2:
        # We are dealing with Covariance matrices
        keys = [(k[0], k[1], k[4], k[5]) for k in keys]
        keys = list(set(keys))
        for key1, key2 in itertools.combinations_with_replacement(keys, 2):
            a1, b1, i1, j1 = key1
            a2, b2, i2, j2 = key2
            covkey = (a1, b1, a2, b2, i1, j1, i2, j2)
            ells1, ells2 = results[covkey].ell
            nells1, nells2 = len(ells1), len(ells2)
            # Writes the covkeys of the spin components associated with covkey
            # it also returns what fields go in what axis
            #  comps1/comp2  (E, E) (B, B) (E,B)
            # (POS, E)
            # (POS, B)
            comps1 = _split_comps(key1)
            comps2 = _split_comps(key2)
            # Save comps in dtype metadata
            dt = np.dtype(
                float,
                metadata={
                    "fields1": comps1,
                    "fields2": comps2,
                },
            )
            _cov = np.zeros((len(comps1), len(comps2), nells1, nells2), dtype=dt)
            for i in range(len(comps1)):
                for j in range(len(comps2)):
                    comp1 = comps1[i]
                    comp2 = comps2[j]
                    _a1, _b1, _, _ = comp1
                    _a2, _b2, _, _ = comp2
                    _covkey = _a1, _b1, _a2, _b2, i1, j1, i2, j2
                    if _covkey not in results.keys():
                        # This triggers if the element doesn't exist
                        # but the symmetrical term does
                        _cov[i, j, :, :] = np.zeros((nells2, nells1))
                    else:
                        _cov[i, j, :, :] = results[_covkey]
            _results[covkey] = Result(_cov, ell=(ells1, ells2))
    return _results


def _split_comps(key):
    _key = copy.deepcopy(key)
    a, b, i, j = _key
    if (a == 'POS') and (b == "SHE"):
        keys = [
            (a, "G_E", i, j),
            (a, "G_B", i, j),
            ]
    elif (a == b == "SHE") and (i == j):
        keys = [
            ("G_E", "G_E", i, j),
            ("G_B", "G_B", i, j),
            ("G_E", "G_B", i, j),
            ]
    elif (a == b == "SHE") and (i != j):
        keys = [
            ("G_E", "G_E", i, j),
            ("G_B", "G_B", i, j),
            ("G_E", "G_B", i, j),
            ("G_E", "G_B", j, i),
            ]
    else:
        keys = [(a, b, i, j)]
    keys = [format_key(k) for k in keys]
    return keys


def _unsplit_comps(key):
    _key = copy.deepcopy(key)
    a, b, i, j = _key
    if a == "G_E" or a == "G_B":
        a = "SHE"
    if b == "G_E" or b == "G_B":
        b = "SHE"
    _key = (a, b, i, j)
    _key = format_key(_key)
    return _key


def format_key(key):
    """
    Produces a Cl key for data maps.
    input:
        key: Cl key
    returns:
        Clkey: Cl key
    """
    _key = copy.deepcopy(key)
    a, b, i, j = _key
    if a == b and i > j:
        i, j = j, i
    elif (b == "POS") or (b == "G_E" and a == "G_B"):
        a, b = b, a
    return (a, b, i, j)
