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
from .result import truncated
from .transforms import cl2corr, corr2cl, transform_cls
from .utils import get_cl

try:
    from copy import replace
except ImportError:
    # Python < 3.13
    from dataclasses import replace


def natural_unmixing(cls, mls, fields, x0=-2, k=50, patch_hole=True, lmax=None):
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
    wmls = transform_cls(mls, fields)
    wmls = correct_correlation(wmls, x0=x0, k=k)
    return _natural_unmixing(cls, wmls, fields, lmax=lmax)


def _natural_unmixing(d, wm, fields, lmax=None):
    """
    Natural unmixing of the data Cl.
    Args:
        d: Data Cl
        wm: mask correlation function
        fields: list of fields
        patch_hole: If True, apply the patch hole correction
    Returns:
        corr_d: Corrected Cl
    """
    corr_d = {}
    masks = {}
    for key, field in fields.items():
        if field.mask is not None:
            masks[key] = field.mask

    for key in d.keys():
        a, b, i, j = key
        m_key = (masks[a], masks[b], i, j)
        _wm = get_cl(m_key, wm)
        _d = d[key]
        s1, s2 = _d.spin
        if lmax is None:
            *_, lmax = _d.shape
        lmax_mask = len(_wm.array)
        # Grab metadata
        dtype = _d.array.dtype
        # pad cls
        _d = np.atleast_2d(_d.array)
        pad_width = [(0, 0)] * _d.ndim  # no padding for other dims
        pad_width[-1] = (0, lmax_mask - lmax)  # pad only last dim
        _d = np.pad(_d, pad_width, mode="constant", constant_values=0)
        if (s1 != 0) and (s2 != 0):
            __d = np.array(
                [
                    np.zeros_like(_d[0, 0]),
                    _d[0, 0],  # EE like spin-2
                    _d[1, 1],  # BB like spin-2
                    np.zeros_like(_d[0, 0]),
                ]
            )
            __id = np.array(
                [
                    np.zeros_like(_d[0, 0]),
                    -_d[0, 1],  # EB like spin-0
                    _d[1, 0],  # EB like spin-0
                    np.zeros_like(_d[0, 0]),
                ]
            )
            # Correct by alpha
            wd = cl2corr(__d.T).T + 1j * cl2corr(__id.T).T
            corr_wd = (wd / _wm).real
            icorr_wd = (wd / _wm).imag
            # Transform back to Cl
            __corr_d = corr2cl(corr_wd.T).T
            __icorr_d = corr2cl(icorr_wd.T).T
            # reorder
            _corr_d = np.zeros_like(_d)
            _corr_d[0, 0] = __corr_d[1]  # EE like spin-2
            _corr_d[1, 1] = __corr_d[2]  # BB like spin-2
            _corr_d[0, 1] = -__icorr_d[1]  # EB like spin-0
            _corr_d[1, 0] = __icorr_d[2]  # EB like spin-0
        else:
            # Treat everything as spin-0
            _corr_d = []
            for cl in _d:
                wd = cl2corr(cl).T
                corr_wd = wd / _wm
                # Transform back to Cl
                __corr_d = corr2cl(corr_wd.T).T
                _corr_d.append(__corr_d[0])
            # remove extra axis
            _corr_d = np.squeeze(_corr_d)
        # Add metadata back
        _corr_d = np.array(list(_corr_d), dtype=dtype)
        corr_d[key] = replace(d[key], array=_corr_d)
    # truncate to lmax
    corr_d = truncated(corr_d, lmax)
    return corr_d


def correct_correlation(wms, x0=-5, k=50):
    """
    Correct correlation functions using a logistic function.
    Args:
        wms: mask correlation functions
        x0: midpoint of the logistic function
        k: steepness of the logistic function
    Returns:
        corrected_wms: corrected mask correlation functions
    """
    corrected_wms = {}
    for key, wm in wms.items():
        wm = wm.array
        x = np.log10(np.abs(wm))
        correction = logistic(x, x0=x0, k=k)
        corrected_array = wm * correction
        corrected_wms[key] = replace(wms[key], array=corrected_array)
    return corrected_wms

def logistic(x, x0=-5, k=50):
    return 1.0 + np.exp(-k * (x - x0))
