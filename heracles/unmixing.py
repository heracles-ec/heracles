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
from .transforms import transform_cls, transform_corrs
from .utils import get_cl

try:
    from copy import replace
except ImportError:
    # Python < 3.13
    from dataclasses import replace


def natural_unmixing(cls, mls, fields, options={}, rtol=0.3):
    """
    Natural unmixing of the data Cl.
    Args:
        cls: Data Cl
        mls: mask Cl
        fields: list of fields
        patch_hole: If True, apply the patch hole correction
    Returns:
        corr_cls: Corrected Cl
    """
    mask_lmax = mls[list(mls.keys())[0]].shape[-1]
    lmax = cls[list(cls.keys())[0]].shape[-1]-1
    wmls = transform_cls(mls)
    wmls = correct_correlation(wmls, options=options, rtol=rtol)
    wcls = transform_cls(cls, lmax_out=mask_lmax)
    return _natural_unmixing(wcls, wmls, fields, lmax=lmax)


def _natural_unmixing(wcls, wmls, fields, lmax=None):
    """
    Natural unmixing of the data Cl.
    Args:
        wcls: data correlation function
        wmls: mask correlation function
        fields: list of fields
        patch_hole: If True, apply the patch hole correction
    Returns:
        corr_cls: Corrected Cl
    """
    corr_wcls = {}
    masks = {}
    for key, field in fields.items():
        if field.mask is not None:
            masks[key] = field.mask
    for key in wcls.keys():
        a, b, i, j = key
        m_key = (masks[a], masks[b], i, j)
        wml = get_cl(m_key, wmls).array
        corr_wcls[key] = replace(wcls[key], array=wcls[key].array / wml)

    corr_cls = transform_corrs(corr_wcls, lmax_out=lmax)
    return corr_cls


def correct_correlation(wms, options={}, rtol=0.3):
    """
    Correct correlation functions using a logistic function.
    Args:
        wms: mask correlation functions
        rtol: relative tolerance for the cutoff
    Returns:
        corrected_wms: corrected mask correlation functions
    """
    corrected_wms = {}
    for key, wm in wms.items():
        if key in options:
            rtol = options[key]
        else:
            rtol = rtol
        wm = wm.array
        cutoff = rtol * np.max(np.abs(wm))
        _wm = wm * logistic(np.log10(abs(wm)), x0=np.log10(cutoff))
        corrected_wms[key] = replace(wms[key], array=_wm)
    return corrected_wms


def logistic(x, x0=-5, k=50):
    return 1.0 + np.exp(-k * (x - x0))
