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
from .transforms import cl2corr, corr2cl

try:
    from copy import replace
except ImportError:
    # Python < 3.13
    from dataclasses import replace


def correct_correlation(wm, options={}, rtol=0.2, smoothing=50):
    """
    Correct the correlation function of the mask to avoid
    dividing by very small numbers during unmixing.
    Args:
        wm: mask correlation functions
        options: dictionary of options for each mask
        rtol: relative tolerance to apply
        smoothing: smoothing parameter for the logistic function
    Returns:
        wm: corrected mask correlation functions
    """
    wm_keys = list(wm.keys())
    corr_wm = {}
    for wm_key in wm_keys:
        if wm_key in list(options.keys()):
            _rtol = options[wm_key].get("rtol", rtol)
            _smoothing = options[wm_key].get("smoothing", smoothing)
        else:
            _rtol = rtol
            _smoothing = smoothing
        _wm = wm[wm_key]
        _tol = _rtol * np.max(abs(_wm))
        _wm *= logistic(
            np.log10(abs(_wm)),
            tol=np.log10(_tol),
            smoothing=_smoothing,
        )
        corr_wm[wm_key] = _wm
    return corr_wm


def natural_unmixing(d, m, fields, options={}, rtol=0.2, smoothing=50):
    """
    Natural unmixing of the data Cl.
    Args:
        d: data cls
        m: mask cls
        fields: list of fields
        patch_hole: If True, apply the patch hole correction
    Returns:
        corr_d: Corrected Cl
    """
    # inverse mapping of masks to fields
    masks = {}
    for key, field in fields.items():
        if field.mask is not None:
            masks[field.mask] = key

    wm = {}
    m_keys = list(m.keys())
    for m_key in m_keys:
        a, b, i, j = m_key
        # Get corresponding mask keys
        a = masks[a]
        b = masks[b]
        # Transform to real space
        _m = m[m_key].array
        _wm = cl2corr(_m).T[0]
        wm[(a, b, i, j)] = _wm

    wm = correct_correlation(
        wm,
        options=options,
        rtol=rtol,
        smoothing=smoothing,
    )
    return _natural_unmixing(d, wm)


def _natural_unmixing(d, wm):
    """
    Natural unmixing of the data Cl.
    Args:
        d: data cls
        m: mask correlation function
        patch_hole: If True, apply the patch hole correction
    Returns:
        corr_d: Corrected Cl
    """
    corr_d = {}
    for key in list(d.keys()):
        ell = d[key].ell
        if ell is None:
            *_, lmax = d[key].shape
        else:
            lmax = ell[-1] + 1
        s1, s2 = d[key].spin
        _d = np.atleast_2d(d[key])
        # Get corresponding mask correlation function
        if key in wm:
            _wm = wm[key]
        else:
            a, b, i, j = key
            if a == b and (a, b, j, i) in wm:
                _wm = wm[(a, b, j, i)]
            elif (b, a, j, i) in wm:
                _wm = wm[(b, a, j, i)]
            else:
                raise KeyError(f"Key {key} not found in mask correlation functions.")

        lmax_mask = len(_wm)
        # pad cls
        pad_width = [(0, 0)] * _d.ndim  # no padding for other dims
        pad_width[-1] = (0, lmax_mask - lmax)  # pad only last dim
        _d = np.pad(_d, pad_width, mode="constant", constant_values=0)
        # Grab metadata
        dtype = d[key].array.dtype
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


def logistic(x, tol=-5, smoothing=50):
    return 1.0 + np.exp(-smoothing * (x - tol))
