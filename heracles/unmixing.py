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
from .result import binned
from .transforms import cl2corr, corr2cl, l2x
from scipy.integrate import cumulative_simpson


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
            fcls = np.array([[fcls_EE, fcls_EB], [fcls_BE, fcls_BB]])

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


def master(t, d, M, ledges=None):
    """
    Master method for unmixing E/B modes
    Args:
        t: theory cl
        d: data cl
        M: mixing matrix
    """
    ft = forwards(t, M)
    if ledges is not None:
        ft = binned(ft, ledges)
        d = binned(d, ledges)
        M = binned(M, ledges)
        for key in M.keys():
            m = M[key]
            ax = m.axis[0]
            m = np.swapaxes(m, ax, ax + 1)
            M[key] = Result(m, axis=M[key].axis)
        M = binned(M, ledges)
    mt = inversion(ft, M)
    md = inversion(d, M)
    return mt, md


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
        _d = np.atleast_2d(d[d_key])
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
        corr_d[d_key] = Result(_corr_d, axis=axis, ell=ell)
    return corr_d


def PolSpice(d, m, mode="minus", patch_hole=True, x0=-2, k=50):
    wm = {}
    m_keys = list(m.keys())
    for m_key in m_keys:
        _m = m[m_key].array
        _wm = cl2corr(_m).T[0]
        if patch_hole:
            _wm *= logistic(np.log10(abs(_wm)), x0=x0, k=k)
        wm[m_key] = _wm
    return _polspice(d, wm, mode=mode)


def _polspice(d, wm, mode="minus"):
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
        _d = np.atleast_2d(d[d_key])
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
            xi_p = _wm[1].real
            xi_m = _wm[2].real
            x = l2x(ell)
            if mode == "plus":
                xi_dec_plus = Eq90_plus(x, xi_p)
                pols_plus_corrs_1 = np.array([
                    np.zeros_like(_d[0, 0]),
                    np.zeros_like(_d[0, 0]),
                    xi_dec_plus/_wm,
                    np.zeros_like(_d[0, 0])])
                pols_plus_1_cls_list = corr2cl(pols_plus_corrs_1.T).T
                pols_plus_corrs_2 = np.array([
                    np.zeros_like(_d[0, 0]),
                    np.zeros_like(_d[0, 0]),
                    xi_m/_wm,
                    np.zeros_like(_d[0, 0])])
                pols_plus_2_cls_list = corr2cl(pols_plus_corrs_2.T).T
                # reorder
                _corr_d = np.zeros_like(_d)
                _corr_d[0, 0] = -(pols_plus_1_cls_list[2]+pols_plus_2_cls_list[2])  # EE like spin-2
                _corr_d[1, 1] = (pols_plus_1_cls_list[2]-pols_plus_2_cls_list[2])  # BB like spin-2
            elif mode == "minus":
                xi_dec_minus = Eq90_minus(x, xi_m)
                pols_minus_corrs_1 = np.array([
                    np.zeros_like(_d[0, 0]),
                    xi_p/_wm,
                    np.zeros_like(_d[0, 0]),
                    np.zeros_like(_d[0, 0])])
                pols_minus_1_cls_list = corr2cl(pols_minus_corrs_1.T).T
                pols_minus_corrs_2 = np.array([
                    np.zeros_like(_d[0, 0]),
                    xi_dec_minus/_wm,
                    np.zeros_like(_d[0, 0]),
                    np.zeros_like(_d[0, 0])])
                pols_minus_2_cls_list = corr2cl(pols_minus_corrs_2.T).T
                # reorder
                _corr_d = np.zeros_like(_d)
                _corr_d[0, 0] = (pols_minus_1_cls_list[2]+pols_minus_2_cls_list[2])  # EE like spin-2
                _corr_d[1, 1] = (pols_minus_1_cls_list[2]-pols_minus_2_cls_list[2])  # BB like spin-2
            # off-diagonal terms
            icorr_wd = (wd / _wm).imag
            # Transform back to Cl
            __icorr_d = corr2cl(icorr_wd.T).T
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
        corr_d[d_key] = Result(_corr_d, axis=axis, ell=ell)
    return corr_d


def Eq90_plus(x, xi_p):
    theta = np.arccos(x)[::-1]
    dtheta = theta[1] - theta[0]
    prefac1 = 8 * (2 + x) / (1 - x) ** 2
    integ1 = (1 - x) / (1 + x) ** 2
    integ1 *= xi_p
    int1 = cumulative_simpson(integ1[::-1] * np.sin(theta[::-1]), dx=dtheta, initial=0)[::-1]
    t1 = prefac1 * int1
    prefac2 = 8 / (1 - x)
    integ2 = 1 / (1 + x) ** 2
    integ2 *= xi_p
    int2 = cumulative_simpson(integ2[::-1] * np.sin(theta[::-1]), dx=dtheta, initial=0)[::-1]
    t2 = prefac2 * int2
    eq90 = xi_p - t1 + t2
    return eq90


def Eq90_minus(x, xi_m):
    theta = np.arccos(x)[::-1]
    dtheta = theta[1] - theta[0]
    prefac1 = 8 * (2 - x) / (1 + x) ** 2
    integ1 = (1 + x) / (1 - x) ** 2
    integ1 *= xi_m
    int1 = cumulative_simpson(integ1 * np.sin(theta[::-1]), dx=dtheta, initial=0)
    t1 = prefac1 * int1
    prefac2 = 8 / (1 + x)
    integ2 = 1 / (1 - x) ** 2
    integ2 *= xi_m
    int2 = cumulative_simpson(integ2 * np.sin(theta[::-1]), dx=dtheta, initial=0)
    t2 = prefac2 * int2
    eq90 = xi_m - t1 + t2
    return eq90


def logistic(x, x0=-5, k=50):
    return 1.0 + np.exp(-k * (x - x0))
