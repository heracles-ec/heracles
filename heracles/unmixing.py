# Heracles: Euclid code for harmonic-space statistics on the sphere
#
# Copyright (C) 2023-2024 Euclid Science Ground Segment
#
# This file is part of Heracles.
#
# Heracles is free software: you can redistribute it and/or modify it
# under the terms of the GNU Lesser General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Heracles is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public
# License along with Heracles. If not, see <https://www.gnu.org/licenses/>.
import numpy as np
from .result import binned
from .transforms import cl2corr, corr2cl
from .utils import get_cl

try:
    from copy import replace
except ImportError:
    # Python < 3.13
    from dataclasses import replace


def naturalspice(d, m, fields, rcond=0.01):
    """
    Natural unmixing of the data Cl.
    Args:
        d: Data Cl
        m: mask Cl
        fields: list of fields
        patch_hole: If True, apply the patch hole correction
    Returns:
        corr_d: Corrected Cl
    """
    wd = cl2corr(d)
    wm = cl2corr(m)
    for m_key in list(wm.keys()):
        _wm = wm[m_key].array
        _wm = _wm * logistic(np.log10(abs(_wm)), x0=np.log10(rcond * np.max(_wm)))
        wm[m_key] = replace(wm[m_key], array=_wm)
    return _naturalspice(wd, wm, fields)


def _naturalspice(wd, wm, fields):
    """
    Natural unmixing of the data correlation function.
    Args:
        wd: data correlation function
        wm: mask correlation function
        fields: list of fields
        patch_hole: If True, apply the patch hole correction
    Returns:
        corr_d: Corrected Cl
    """
    lmax = len(wd[list(wd.keys())[0]][:])
    lmax_mask = len(wm[list(wm.keys())[0]][:])

    masks = {}
    for key, field in fields.items():
        if field.mask is not None:
            masks[key] = field.mask

    # pad correlation functions to lmax_mask
    wd = binned(wd, np.arange(0, lmax_mask + 1))

    corr_wd = {}
    for key in wd.keys():
        a, b, i, j = key
        m_key = (masks[a], masks[b], i, j)
        _wm = get_cl(m_key, wm)
        _wd = wd[key]
        # divide by the mask correlation function
        corr_wd[key] = replace(wd[key], array=(_wd.array / _wm.array))

    # trnasform back to Cl
    corr_d = corr2cl(corr_wd)

    # truncate to lmax
    corr_d = binned(corr_d, np.arange(0, lmax + 1))
    return corr_d


def logistic(x, x0=-5, k=50):
    return 1.0 + np.exp(-k * (x - x0))
