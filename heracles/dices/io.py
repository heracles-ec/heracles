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
import numpy as np
from ..result import Result


def Fields2Components(result):
    """
    Separates the SHE values into E and B modes.
    input:
        Cls: dictionary of Cl values
    returns:
        Cls_unraveled: dictionary of Cl values
    """
    _result = {}
    for key in list(result.keys()):
        t1, t2, b1, b2 = key
        _r = result[key]
        ell = _r.ell
        if t1 == t2 == "SHE" and b1 == b2:
            _result[("G_E", "G_E", b1, b2)] = Result(_r[..., 0, :], ell)
            _result[("G_B", "G_B", b1, b2)] = Result(_r[..., 1, :], ell)
            _result[("G_E", "G_B", b1, b2)] = Result(_r[..., 2, :], ell)
        elif t1 == t2 == "SHE" and b1 != b2:
            _result[("G_E", "G_E", b1, b2)] = Result(_r[..., 0, :], ell)
            _result[("G_B", "G_B", b1, b2)] = Result(_r[..., 1, :], ell)
            _result[("G_E", "G_B", b1, b2)] = Result(_r[..., 2, :], ell)
            _result[("G_E", "G_B", b2, b1)] = Result(_r[..., 3, :], ell)
        elif t1 == "POS" and t2 == "SHE":
            _result[("POS", "G_E", b1, b2)] = Result(_r[..., 0, :], ell)
            _result[("POS", "G_B", b1, b2)] = Result(_r[..., 1, :], ell)
        else:
            _result[key] = _r
    return _result


def Components2Data(cls, cov):
    Clkeys = list(cls.keys())
    nells = [len(cls[key].ell) for key in Clkeys]
    full_cov = np.zeros((np.sum(nells), np.sum(nells)))
    for i in range(0, len(Clkeys)):
        for j in range(i, len(Clkeys)):
            ki = Clkeys[i]
            kj = Clkeys[j]
            A, B, nA, nB = ki[0], ki[1], ki[2], ki[3]
            C, D, nC, nD = kj[0], kj[1], kj[2], kj[3]
            covkey = (A, B, C, D, nA, nB, nC, nD)
            size_i = nells[i]
            size_j = nells[j]
            full_cov[
                i * size_i : (i + 1) * size_i, j * size_j : (j + 1) * size_j
            ] = cov[covkey]
            if i != j:
                full_cov[
                    j * size_j : (j + 1) * size_j, i * size_i : (i + 1) * size_i
                ] = cov[covkey].T
    return full_cov


def Data2Components(cls, cov):
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


def Components2Fields(cls, cov):
    _covs = {}
    cls_keys = list(cls.keys())
    for i in range(0, len(cls_keys)):
        for j in range(i, len(cls_keys)):
            k1 = cls_keys[i]
            k2 = cls_keys[j]
            ell1 = cls[k1].ell
            ell2 = cls[k2].ell
            cl1 = np.atleast_2d(cls[k1])
            cl2 = np.atleast_2d(cls[k2])
            ncls1, nells1 = cl1.shape
            ncls2, nells2 = cl2.shape
            A, B, nA, nB = k1[0], k1[1], k1[2], k1[3]
            C, D, nC, nD = k2[0], k2[1], k2[2], k2[3]

            covkey = (A, B, C, D, nA, nB, nC, nD)
            # Writes the covkeys of the spin components associated with covkey
            # it also returns what fields go in what axis
            #  comps1     (E, E) (B, B) (E,B) <--- comps2
            # (POS, E)
            # (POS, B)
            covkeys, comps1, comps2 = _split_comps(covkey, ncls1, ncls2)
            # Save comps in dtype metadata
            dt = np.dtype(
                float,
                metadata={
                    "fields1": comps1,
                    "fields2": comps2,
                },
            )
            _cov = np.zeros((ncls1, ncls2, nells1, nells2), dtype=dt)
            for i in range(ncls1):
                for j in range(ncls2):
                    _covkey = covkeys[(i, j)]
                    if _covkey not in cov.keys():
                        # This triggers if the element doesn't exist
                        # but the symmetrical term does
                        _cov[i, j, :, :] = np.zeros((nells2, nells1))
                    else:
                        _cov[i, j, :, :] = cov[_covkey]
            _covs[covkey] = Result(_cov, ell=(ell1, ell2))
    return _covs


def _split_comps(covkey, ncls1, ncls2):
    a1, b1, a2, b2, i1, j1, i2, j2 = covkey
    if ncls1 == 1:
        f1 = [("POS", "POS")]
    elif ncls1 == 2:
        f1 = [("POS", "G_E"), ("POS", "G_B")]
    elif ncls1 == 3:
        f1 = [("G_E", "G_E"), ("G_B", "G_B"), ("G_E", "G_B")]
    elif ncls2 == 4:
        f1 = [("G_E", "G_E"), ("G_B", "G_B"), ("G_E", "G_B"), ("G_B", "G_E")]

    if ncls2 == 1:
        f2 = [("POS", "POS")]
    elif ncls2 == 2:
        f2 = [("POS", "G_E"), ("POS", "G_B")]
    elif ncls2 == 3:
        f2 = [("G_E", "G_E"), ("G_B", "G_B"), ("G_E", "G_B")]
    elif ncls2 == 4:
        f2 = [("G_E", "G_E"), ("G_B", "G_B"), ("G_E", "G_B"), ("G_B", "G_E")]

    covkeys = {}
    for i in range(ncls1):
        for j in range(ncls2):
            _f1 = f1[i]
            _f2 = f2[j]
            _a1, _b1 = _f1
            _a2, _b2 = _f2
            if _a2 == "G_B" and _b2 == "G_E":
                _a2, _b2 = "G_E", "G_B"
                i2, j2 = j2, i2
            _covkey = _a1, _b1, _a2, _b2, i1, j1, i2, j2
            covkeys[(i, j)] = _covkey
    return covkeys, f1, f2