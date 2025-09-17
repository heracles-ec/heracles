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
from .result import Result, truncated
from .transforms import cl2corr, corr2cl


def natural_unmixing(d, m, patch_hole=True, x0=-2, k=50, lmax=None):
    wm = {}
    m_keys = list(m.keys())
    for m_key in m_keys:
        _m = m[m_key].array
        _wm = cl2corr(_m).T[0]
        if patch_hole:
            _wm *= logistic(np.log10(abs(_wm)), x0=x0, k=k)
        wm[m_key] = _wm
    return _natural_unmixing(d, wm, lmax=lmax)


def _natural_unmixing(d, wm, lmax=None):
    """
    Natural unmixing of the data Cl.
    Args:
        d: Data Cl
        m: mask cls
        patch_hole: If True, apply the patch hole correction
    Returns:
        corr_d: Corrected Cl
    """
    corr_d = {}
    d_keys = list(d.keys())
    wm_keys = list(wm.keys())
    for d_key, wm_key in zip(d_keys, wm_keys):
        a, b, i, j = d_key
        if lmax is None:
            *_, lmax = d[d_key].shape
        _d = np.atleast_2d(d[d_key])
        lmax_mask = len(wm[wm_key])
        # pad cls
        pad_width = [(0, 0)] * _d.ndim  # no padding for other dims
        pad_width[-1] = (0, lmax_mask - lmax)  # pad only last dim
        _d = np.pad(_d, pad_width, mode="constant", constant_values=0)
        # invert mask
        _wm = wm[wm_key]
        _inv_wm = 1.0 / _wm
        # Grab metadata
        dtype = d[d_key].array.dtype
        axis = d[d_key].axis
        ell = np.arange(lmax + 1)
        if a == b == "SHE":
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
            corr_wd = (wd * _inv_wm).real
            icorr_wd = (wd * _inv_wm).imag
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
                corr_wd = wd * _inv_wm
                # Transform back to Cl
                __corr_d = corr2cl(corr_wd.T).T
                _corr_d.append(__corr_d[0])
            # remove extra axis
            _corr_d = np.squeeze(_corr_d)
        # Add metadata back
        _corr_d = np.array(list(_corr_d), dtype=dtype)
        corr_d[d_key] = Result(_corr_d, axis=axis, ell=ell)
    # truncate to lmax
    corr_d = truncated(corr_d, lmax)
    return corr_d


def logistic(x, x0=-5, k=50):
    return 1.0 + np.exp(-k * (x - x0))
