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
                return np.transpose(arr, axes=(1, 0, 2))
            else:
                return arr

        else:
            raise KeyError(f"Key {key} not found in Cls.")


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
        cov_c[key] = replace(a, array=c)
    return cov_c


def _flatten(result):
    a = result.array
    axis = len(result.axis)
    if axis == 1:
        s1, s2 = result.spin
        dof1 = 1 if s1 == 0 else 2
        dof2 = 1 if s2 == 0 else 2
        ell = a.shape[-1]
        b = a.reshape(dof1 * dof2, ell)
        b = b.transpose(0, 1).reshape(dof1 * dof2 * ell)
    elif axis == 2:
        s1, s2, s3, s4 = result.spin
        dof1 = 1 if s1 == 0 else 2
        dof2 = 1 if s2 == 0 else 2
        dof3 = 1 if s3 == 0 else 2
        dof4 = 1 if s4 == 0 else 2
        ell = a.shape[-1]
        b = (
            a.reshape(dof1 * dof2, dof3 * dof4, ell, ell)
            .transpose(0, 2, 1, 3)
            .reshape(dof1 * dof2 * ell, dof3 * dof4 * ell)
        )
    else:
        raise NotImplementedError("Flattening for >2 axes not implemented yet.")
    return b


def flatten(results, order=None):
    # Flatten each block
    blocks_dict = {}
    for key, result in results.items():
        blocks_dict[key] = _flatten(result)

    # check that all results have the same length axis
    axis = [len(result.axis) for result in results.values()]
    axis = np.unique(axis)
    if len(axis) != 1:
        raise ValueError("All results must have the same length axis to flatten.")
    else:
        axis = axis[0]

    if axis == 1:
        # Stack all blocks vertically
        return np.vstack(list(blocks_dict.values()))
    elif axis == 2:
        # Infer order if not provided
        if order is None:
            keys = list(blocks_dict.keys())
            keys = [(key[0], key[1], key[4], key[5]) for key in keys]
            order = list(set(keys))

        # Build row by row
        block_rows = []
        for key_i in order:
            row_blocks = []
            for key_j in order:
                a1, b1, i1, j1 = key_i
                a2, b2, i2, j2 = key_j
                cov_key = (a1, b1, a2, b2, i1, j1, i2, j2)
                block = blocks_dict.get((cov_key))
                if block is None:
                    # If only the transpose block exists, use it transposed
                    sym_cov_key = (a2, b2, a1, b1, i2, j2, i1, j1)
                    transpose_block = blocks_dict.get((sym_cov_key))
                    if transpose_block is not None:
                        block = transpose_block.T
                    else:
                        raise KeyError(f"Missing block for {cov_key}")
                row_blocks.append(block)
            block_rows.append(row_blocks)

        # Use np.block to assemble the full covariance matrix
        return np.block(block_rows)
    else:
        raise NotImplementedError("Flattening for axis != 2 not implemented yet.")
