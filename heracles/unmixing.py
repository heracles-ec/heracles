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
from .transforms import _cached_gauss_legendre

try:
    from copy import replace
except ImportError:
    # Python < 3.13
    from dataclasses import replace


def logistic(x, x0=-2, k=50):
    return 1.0 + np.exp(-k * (x - x0))


def naturalspice(d, m, fields, theta_max=None):
    """
    Natural unmixing of the data Cl.
    Args:
        d: Data Cl
        m: mask Cl
        fields: list of fields
        theta_max: maximum angle to use for the unmixing, in degrees. If None, use all angles.
    Returns:
        corr_d: Corrected Cl
    """
    first_wd = list(d.values())[0]
    first_wm = list(m.values())[0]
    lmax = first_wd.shape[first_wd.axis[0]]
    lmax_mask = first_wm.shape[first_wm.axis[0]]

    # pad correlation functions to lmax_mask
    d = binned(d, np.arange(0, lmax_mask + 1))

    wd = cl2corr(d)
    wm = cl2corr(m)
    corr_wds = _naturalspice(wd, wm, fields, theta_max=theta_max)

    # trnasform back to Cl
    corr_d = corr2cl(corr_wds)

    # truncate to lmax
    corr_d = binned(corr_d, np.arange(0, lmax + 1))
    return corr_d


def _naturalspice(wd, wm, fields, theta_max=None):
    """
    Natural unmixing of the data correlation function.
    Args:
        wd: data correlation function
        wm: mask correlation function
        fields: list of fields
        theta_max: maximum angle in degrees for the logistic cutoff. If None, uses default x0=-2.
    Returns:
        corr_d: Corrected Cl
    """
    masks = {}
    for key, field in fields.items():
        if field.mask is not None:
            masks[key] = field.mask

    if theta_max is not None:
        first_wm = list(wm.values())[0]
        lmax_mask = first_wm.shape[first_wm.axis[0]]
        xvals, _ = _cached_gauss_legendre(lmax_mask)
        theta = np.arccos(xvals) * 180 / np.pi
        i_theta_max = np.abs(theta - theta_max).argmin()
        x0 = np.log10(abs(first_wm[i_theta_max]))
    else:
        x0 = -5

    corr_wds = {}
    for key in wd.keys():
        a, b, i, j = key
        m_key = (masks[a], masks[b], i, j)
        _wm = get_cl(m_key, wm).array
        _wd = wd[key].array
        _wm *= logistic(np.log10(abs(_wm)), x0=x0)
        corr_wds[key] = replace(wd[key], array=(_wd / _wm))

    return corr_wds
