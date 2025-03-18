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
from copy import deepcopy
from ..fields import Positions, Shears, Visibility, Weights
from ..mapping import transform
from ..result import Result
from ..healpy import HealpixMapper
from ..twopoint import angular_power_spectra


def get_cls(maps, jkmaps, fields, jk=0, jk2=0):
    """
    Internal method to compute the Cls of removing 2 Jackknife.
    inputs:
        maps (dict): Dictionary of data maps
        jkmaps (dict): Dictionary of mask maps
        fields (dict): Dictionary of fields
        jk (int): Jackknife region to remove
        jk2 (int): Jackknife region to remove
    returns:
        cls (dict): Dictionary of data Cls
    """
    # grab metadata
    print(f" - Computing Cls for regions ({jk},{jk2})", end="\r", flush=True)
    _m = maps[list(maps.keys())[0]]
    meta = _m.dtype.metadata
    lmax = meta["lmax"]
    ell = np.arange(lmax + 1)
    # deep copy to avoid modifying the original maps
    _maps = deepcopy(maps)
    for key_data, key_mask in zip(maps.keys(), jkmaps.keys()):
        _map = _maps[key_data]
        _jkmap = jkmaps[key_mask]
        _mask = np.copy(_jkmap)
        _mask[_mask != 0] = _mask[_mask != 0] / _mask[_mask != 0]
        # Remove jk 2 regions
        cond = np.where((_jkmap == float(jk)) | (_jkmap == float(jk2)))[0]
        _mask[cond] = 0.0
        # Apply mask
        _map *= _mask
    # compute alms
    alms = transform(fields, _maps)
    # compute cls
    cls = angular_power_spectra(alms)
    # Result
    for key in cls.keys():
        cls[key] = Result(cls[key], ell=ell)
    return cls
