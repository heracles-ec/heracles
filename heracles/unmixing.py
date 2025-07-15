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
from .transforms import cl2corr, corr2cl, _cached_gauss_legendre
from scipy.integrate import cumulative_simpson


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
        if a == b == "SHE":
            _d = np.array(
                [
                    np.zeros(len(ell)),
                    d[d_key][0, 0],
                    d[d_key][1, 1],
                    np.zeros(len(ell)),
                ]
            )
            _id = np.array(
                [
                    np.zeros(len(ell)),
                    -d[d_key][0, 1],
                    d[d_key][1, 0],
                    np.zeros(len(ell)),
                ]
            )
            # Transform to correlation space
            wd = cl2corr(_d.T).T + 1j * cl2corr(_id.T).T
            xi_p = wd[1].real
            xi_m = wd[2].real
            corr_x, _ = _cached_gauss_legendre(ell[-1] + 1)
            if mode == "plus":
                xi_dec_plus = Eq90_plus(corr_x, xi_p) / _wm
                pp1_corrs = np.array(
                    [
                        np.zeros(len(ell)),
                        np.zeros(len(ell)),
                        xi_dec_plus,
                        np.zeros(len(ell)),
                    ]
                )
                pp1_cls = corr2cl(pp1_corrs.T).T
                pp2_corrs = np.array(
                    [
                        np.zeros(len(ell)),
                        np.zeros(len(ell)),
                        xi_m / _wm,
                        np.zeros(len(ell)),
                    ]
                )
                pp2_cls = corr2cl(pp2_corrs.T).T
                # reorder
                _corr_d = np.zeros_like(d[d_key])
                _corr_d[0, 0] = -(pp1_cls[2] + pp2_cls[2])
                _corr_d[1, 1] = pp1_cls[2] - pp2_cls[2]
            elif mode == "minus":
                xi_dec_minus = Eq90_minus(corr_x, xi_m) / _wm
                pm1_corrs = np.array(
                    [
                        np.zeros(len(ell)),
                        xi_p / _wm,
                        np.zeros(len(ell)),
                        np.zeros(len(ell)),
                    ]
                )
                pm1_cls = corr2cl(pm1_corrs.T).T
                pm2_corrs = np.array(
                    [
                        np.zeros(len(ell)),
                        xi_dec_minus,
                        np.zeros(len(ell)),
                        np.zeros(len(ell)),
                    ]
                )
                pm2_cls = corr2cl(pm2_corrs.T).T
                # reorder
                _corr_d = np.zeros_like(d[d_key])
                _corr_d[0, 0] = pm1_cls[1] + pm2_cls[1]
                _corr_d[1, 1] = pm1_cls[1] - pm2_cls[1]
            # off-diagonal terms
            icorr_wd = (wd / _wm).imag
            # Transform back to Cl
            __icorr_d = corr2cl(icorr_wd.T).T
            _corr_d[0, 1] = -__icorr_d[1]  # EB like spin-0
            _corr_d[1, 0] = __icorr_d[2]  # EB like spin-0
        else:
            # Treat everything as spin-0
            _d = np.atleast_2d(d[d_key])
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
    int1 = cumulative_simpson(integ1[::-1] * np.sin(theta[::-1]), dx=dtheta, initial=0)[
        ::-1
    ]
    t1 = prefac1 * int1
    prefac2 = 8 / (1 - x)
    integ2 = 1 / (1 + x) ** 2
    integ2 *= xi_p
    int2 = cumulative_simpson(integ2[::-1] * np.sin(theta[::-1]), dx=dtheta, initial=0)[
        ::-1
    ]
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


def logistic(x, x0=-2, k=50):
    return 1.0 + np.exp(-k * (x - x0))
