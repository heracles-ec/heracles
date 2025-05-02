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
from .dices.mask_correction import cl2corr, corr2cl, logistic


def forwards(t, M):
    """
    Forward model for the unmixing E/B modes.
    Args:
        t: Theory Cl
        M: Mixing matrix
    Returns:
        forward_cls: partial sky Cl
    """
    forward_cls = {}
    for key in list(t.keys()):
        a, b, i, j = key
        _t = np.atleast_2d(t[key])
        _M = M[key].array
        if a == b == "SHE":
            # Cl_EE = M_EE Cl_EE + M_BB Cl_BB
            fcls_EE = _M[0] @ _t[0, 0, :] + _M[1] @ _t[1, 1, :]
            # Cl_BB = M_EE Cl_BB + M_BB Cl_EE
            fcls_BB = _M[0] @ _t[1, 1, :] + _M[1] @ _t[0, 0, :]
            # Cl_EB = M_EB Cl_EB
            fcls_EB = _M[2] @ _t[0, 1, :]
            # Cl_BE = M_EB Cl_BE
            fcls_BE = _M[2] @ _t[1, 0, :]
            fcls = np.array(
                [[fcls_EE, fcls_EB],
                [fcls_BE, fcls_BB]])

        else:
            fcls = np.array([_M @ __t for __t in _t])
        *_, m = fcls.shape
        # Check if fcls is a 1D array
        if len(fcls) == 1:
            fcls = fcls[0]
        forward_cls[key] = Result(fcls, axis=t[key].axis, ell=t[key].ell[:m])
    return forward_cls


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
            _M_EE = np.hstack((_M[0], _M[2]))
            _M_BB = np.hstack((_M[2], _M[1]))
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
            _id = np.array([[_id_EE, _id_EB],
                           [_id_BE, _id_BB]])
        else:
            _inv_M = np.linalg.pinv(_M)
            _id = np.array([_inv_M @ __d.T for __d in _d])
            _id = _id[:, :_n]
        if len(_id) == 1:
            _id = _id[0]
        inversion_cls[key] = Result(_id, axis=d[key].axis, ell=d[key].ell)
    return inversion_cls


def natural_unmixing(d, m, patch_hole=True):
    """
    Natural unmixing of the data Cl.
    Args:
        d: Data Cl
        m: Mixing matrix
        patch_hole: If True, apply the patch hole correction
    Returns:
        corr_d: Corrected Cl
    """
    corr_d = {}
    d_keys = list(d.keys())
    m_keys = list(m.keys())
    for d_key, m_key in zip(d_keys, m_keys):
        a, b, i, j = d_key
        _d = np.atleast_2d(d[d_key])
        _m = m[m_key]
        # Grab metadata
        dtype = d[d_key].array.dtype
        ell = d[d_key].ell
        axis = d[d_key].axis
        # transform mask
        __m = np.array(
            [
                _m,
                np.zeros_like(_m),
                np.zeros_like(_m),
                np.zeros_like(_m),
            ]
        )
        wm = cl2corr(__m.T).T[0]
        if patch_hole:
            wm /= logistic(np.log10(abs(wm)), x0=-2, k=50)
        # Correct Cl by mask
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
            corr_wd = (wd / wm).real
            icorr_wd = (wd / wm).imag
            # Transform back to Cl
            __corr_d = corr2cl(corr_wd.T).T
            __icorr_d = corr2cl(icorr_wd.T).T
            # reorder
            _corr_d = np.zeros_like(_d)
            _corr_d[0, 0] = __corr_d[0]  # EE like spin-2
            _corr_d[1, 1] = __corr_d[1]  # BB like spin-2
            _corr_d[0, 1] = -__icorr_d[0]  # EB like spin-0
            _corr_d[1, 0] = __icorr_d[1]  # EB like spin-0
        else:
            # Treat everything as spin-0
            _corr_d = []
            for cl in _d:
                __d = np.array(
                    [
                        cl,
                        np.zeros_like(cl),
                        np.zeros_like(cl),
                        np.zeros_like(cl),
                    ]
                )
                wd = cl2corr(__d.T).T
                corr_wd = wd / wm
                # Transform back to Cl
                __corr_d = corr2cl(corr_wd.T).T
                _corr_d.append(__corr_d[0])

        # remove extra axis
        _corr_d = np.squeeze(_corr_d)
        # Add metadata back
        _corr_d = np.array(_corr_d, dtype=dtype)
        corr_d[d_key] = Result(_corr_d, axis=axis, ell=ell)
    return corr_d
