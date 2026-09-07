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
from .progress import NoProgress, Progress
from .result import binned, get_result_array
from .transforms import cl2corr, corr2cl, _corr2cl
from .transforms import purify as _purify_xi_plus
from .utils import get_cl
from .transforms import _cached_gauss_legendre

try:
    from copy import replace
except ImportError:
    # Python < 3.13
    from dataclasses import replace


def logistic(x, x0=-2, k=20):
    return 1.0 + np.exp(-k * (x - x0))


def naturalspice(d, m, fields, theta_max=None, purify=False, progress: Progress | None = None):
    """
    Natural unmixing of the data Cl.
    Args:
        d: Data Cl
        m: mask Cl
        fields: list of fields
        theta_max: maximum angle to use for the unmixing, in degrees. If None, use all angles.
        purify: whether to purify the EE/BB estimator (only affects s1=s2=2 fields)
        progress: optional progress reporter
    Returns:
        corr_d: Corrected Cl
    """
    if progress is None:
        progress = NoProgress()

    first_wd = list(d.values())[0]
    first_wm = list(m.values())[0]
    lmax = first_wd.shape[first_wd.axis[0]]
    lmax_mask = first_wm.shape[first_wm.axis[0]]

    # pad correlation functions to lmax_mask
    d = binned(d, np.arange(0, lmax_mask + 1))

    with progress.task("data correlations") as task:
        wd = cl2corr(d, progress=task)
    with progress.task("mask correlations") as task:
        wm = cl2corr(m, progress=task)
    with progress.task("unmixing") as task:
        corr_wd = _naturalspice(wd, wm, fields, theta_max=theta_max, progress=task)

    # trnasform back to Cl
    if purify:
        # For s1=s2=2 (EE/BB) keys, use the delta-function correction
        # `purify` to turn the unmixed Xi^+ = corr_wd[key][0, 0] into the
        # "dec" correlation Xi^+_dec = purify(Xi^+), then build the pure
        # EE/BB correlation functions
        #     Xi^EE = Xi^+_dec + Xi^-
        #     Xi^BB = Xi^+ - Xi^+_dec
        # which are transformed to Cl with the *opposite* Wigner matrix
        # from the one they would normally use (Xi^EE via d^l_{2,-2},
        # Xi^BB via d^l_{2,2}):
        #     Cl^EE = int 1/2 (Xi^+_dec + Xi^-) d^l_{2,-2}
        #     Cl^BB = int 1/2 (Xi^+ - Xi^+_dec) d^l_{2,2}
        # All other keys (TE, TT) are unaffected by purification and go
        # through the ordinary corr2cl.
        with progress.task("purified transform back to Cl") as task:
            spin2_keys = [
                key for key, wd in corr_wd.items() if wd.spin[0] != 0 and wd.spin[1] != 0
            ]
            other = {key: wd for key, wd in corr_wd.items() if key not in spin2_keys}
            corr_d = corr2cl(other) if other else {}

            current, total = 0, len(spin2_keys)
            for key in spin2_keys:
                current += 1
                task.update(current, total)

                wd = corr_wd[key]
                dtype = wd.array.dtype
                xvals = get_result_array(wd, "ell")[0]
                theta = np.degrees(np.arccos(xvals))
                key_lmax = len(xvals) - 1

                Xi_p, Xi_m = wd[0, 0], wd[1, 1]
                Xi_p_dec = _purify_xi_plus(Xi_p, theta, theta_max=theta_max)

                n = len(theta)
                corr_BB = np.zeros((2, 2, n))
                corr_BB[0, 0] = Xi_p - Xi_p_dec  # -> Cl^BB via d^l_{2,2}
                corr_EE = np.zeros((2, 2, n))
                corr_EE[1, 1] = Xi_p_dec + Xi_m  # -> Cl^EE via d^l_{2,-2}
                # EB cross-term: unaffected by purification
                corr_eb = np.zeros((2, 2, n))
                corr_eb[0, 1] = wd[0, 1]
                corr_eb[1, 0] = wd[1, 0]

                cl_BB = _corr2cl(corr_BB, (2, 2), lmax=key_lmax)[0, 0]
                cl_EE = _corr2cl(corr_EE, (2, 2), lmax=key_lmax)[0, 0]
                cl_eb = _corr2cl(corr_eb, (2, 2), lmax=key_lmax)

                cl = np.zeros_like(wd.array)
                cl[0, 0] = cl_EE
                cl[1, 1] = cl_BB
                cl[0, 1] = cl_eb[0, 1]
                cl[1, 0] = cl_eb[1, 0]
                cl = np.array(list(cl), dtype=dtype)

                corr_d[key] = replace(wd, ell=np.arange(key_lmax + 1), array=cl)
    else:
        with progress.task("transform back to Cl") as task:
            corr_d = corr2cl(corr_wd, progress=task)

    # truncate to lmax
    corr_d = binned(corr_d, np.arange(0, lmax + 1))
    return corr_d


def _naturalspice(wd, wm, fields, theta_max=None, progress: Progress | None = None):
    """
    Natural unmixing of the data correlation function.
    Args:
        wd: data correlation function
        wm: mask correlation function
        fields: list of fields
        theta_max: maximum angle in degrees for the logistic cutoff. If None, uses default x0=-2.
        progress: optional progress reporter
    Returns:
        corr_d: Corrected Cl
    """
    if progress is None:
        progress = NoProgress()

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
    current, total = 0, len(wd)
    for key in wd.keys():
        current += 1
        progress.update(current, total)
        a, b, i, j = key
        m_key = (masks[a], masks[b], i, j)
        _wm = get_cl(m_key, wm).array
        _wd = wd[key].array
        _wm *= logistic(np.log10(abs(_wm)), x0=x0)
        corr_wds[key] = replace(wd[key], array=(_wd / _wm))

    return corr_wds
