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
from .result import Result
from .transforms import cl2corr, corr2cl


def inversion(d, M):
    """
    Inversion model for the unmixing E/B modes.
    Args:
        d: Data Cl
        M: Mixing matrix
        Returns:
        inversion_cls: inverted Cl
    """
    inversion_cls = {}
    for key in list(d.keys()):
        a, b, i, j = key
        _d = np.atleast_2d(d[key])
        _M = M[key].array
        *_, _n, _m = _M.shape
        if a == b == "SHE":
            _M_EB = _M[2]
            _M_EE = np.hstack((_M[0], _M[1]))
            _M_BB = np.hstack((_M[1], _M[0]))
            _M_EEBB = np.vstack((_M_EE, _M_BB))
            _inv_M_EEBB = np.linalg.pinv(_M_EEBB)
            _inv_M_EB = np.linalg.pinv(_M_EB)
            _d_EEBB = np.hstack((_d[0, 0, :], _d[1, 1, :]))
            _id_EEBB = _inv_M_EEBB @ _d_EEBB
            _id_EE = _id_EEBB[:_m][:_n]
            _id_BB = _id_EEBB[_m:][:_n]
            _id_EB = _inv_M_EB @ _d[0, 1, :]
            _id_BE = _inv_M_EB @ _d[1, 0, :]
            _id_EB = _id_EB[:_n]
            _id_BE = _id_BE[:_n]
            _id = np.array([[_id_EE, _id_EB], [_id_BE, _id_BB]])
        else:
            _inv_M = np.linalg.pinv(_M)
            _id = np.array([_inv_M @ __d.T for __d in _d])
            _id = _id[:, :_n]
        if len(_id) == 1:
            _id = _id[0]
        inversion_cls[key] = Result(_id, axis=d[key].axis, ell=d[key].ell)
    return inversion_cls


def natural_unmixing(d, m, patch_hole=True, x0=-2, k=50):
    wm = {}
    m_keys = list(m.keys())
    for m_key in m_keys:
        _m = m[m_key].array
        _wm = cl2corr(_m).T[0]
        if patch_hole:
            _wm *= logistic(np.log10(abs(_wm)), x0=x0, k=k)
        wm[m_key] = _wm
    return _natural_unmixing(d, wm)


def _natural_unmixing(d, wm):
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
        _wm = wm[wm_key]
        # Grab metadata
        dtype = d[d_key].array.dtype
        ell = d[d_key].ell
        axis = d[d_key].axis
        # Check if ell is None
        if ell is None:
            ell = np.arange(len(_wm))
        if a == b == "SHE":
            _d = np.array(
                [
                    np.zeros(len(ell)),
                    d[d_key][0, 0],  # EE like spin-2
                    d[d_key][1, 1],  # BB like spin-2
                    np.zeros(len(ell)),
                ]
            )
            _id = np.array(
                [
                    np.zeros(len(ell)),
                    -d[d_key][0, 1],  # EB like spin-0
                    d[d_key][1, 0],  # EB like spin-0
                    np.zeros(len(ell)),
                ]
            )
            # Correct by alpha
            wd = cl2corr(_d.T).T + 1j * cl2corr(_id.T).T
            corr_wd = (wd / _wm).real
            icorr_wd = (wd / _wm).imag
            # Transform back to Cl
            __corr_d = corr2cl(corr_wd.T).T
            __icorr_d = corr2cl(icorr_wd.T).T
            # reorder
            _corr_d = np.zeros_like(d[d_key])
            _corr_d[0, 0] = __corr_d[1]  # EE like spin-2
            _corr_d[1, 1] = __corr_d[2]  # BB like spin-2
            _corr_d[0, 1] = -__icorr_d[1]  # EB like spin-0
            _corr_d[1, 0] = __icorr_d[2]  # EB like spin-0
        else:
            # Treat everything as spin-0
            _corr_d = []
            _d = np.atleast_2d(d[d_key])
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
        corr_d[d_key] = Result(_corr_d, axis=axis, ell=ell)
    return corr_d


def logistic(x, x0=-2, k=50):
    return 1.0 + np.exp(-k * (x - x0))
