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
from .utils import get_cl
from .transforms import _cached_gauss_legendre

try:
    from copy import replace
except ImportError:
    # Python < 3.13
    from dataclasses import replace


def logistic(x, x0=-2, k=20):
    return 1.0 + np.exp(-k * (x - x0))


def gaussian_apod(theta, theta_max):
    """
    Gaussian apodization window in theta (degrees), matching PolSpice's
    `apodizefunction` type 0 (see apodize_mod.f90 in the PolSpice source):
    `theta_max` is used as both the taper's FWHM and its hard cutoff radius.
    If `theta_max` is None, no apodization is applied (flat weight of 1
    everywhere).
    """
    if theta_max is None:
        return np.ones_like(theta)
    sigma = theta_max / np.sqrt(8 * np.log(2))
    return np.where(theta < theta_max, np.exp(-0.5 * (theta / sigma) ** 2), 0.0)


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
        # For s1=s2=2 (EE/BB) keys, use PolSpice's "decouple" estimator
        # (Chon et al. 2004, eq. 65): instead of the usual pairing
        # (Xi^+ via d^l_{2,2}, Xi^- via d^l_{2,-2}), both Cl^EE and Cl^BB
        # are built from the *same* kernel, d^l_{2,-2}, applied to
        # (Xi^+ + Xi^-) and (Xi^+ - Xi^-) respectively, each normalized by
        # a per-l coupling factor Fl computed from a csc^2(theta/2)-weighted
        # (optionally Gaussian-apodized) quadrature of that same kernel:
        #     Cl^EE = pi * int (Xi^+ + Xi^-) d^l_{2,-2} / Fl
        #     Cl^BB = pi * int (Xi^+ - Xi^-) d^l_{2,-2} / Fl
        #     Fl    =      int apod(theta) csc^2(theta/2) d^l_{2,-2}
        # Each of these is computed by reusing `_corr2cl(..., (2, 2), ...)`
        # on an isolated-slot array: feeding a quantity into the corr[1, 1]
        # ("m_diag", d^l_{2,-2}-kernel) slot and reading back the output EE
        # slot gives exactly pi * int (...) d^l_{2,-2} (the "pi" here is
        # `_corr2cl`'s own 2*pi normalization, halved by `unrotate`), so an
        # extra pi is applied explicitly to cl_EE/cl_BB below to get the
        # single power of pi the formula above calls for once the ratio
        # with Fl (which carries none) has cancelled the other one.
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
                n = len(theta)

                Xi_p, Xi_m = wd[0, 0], wd[1, 1]
                apod = gaussian_apod(theta, theta_max)
                csc2 = 1.0 / np.sin(np.radians(theta) / 2) ** 2

                # feed each quantity into the corr[1, 1] ("m_diag",
                # d^l_{2,-2}-kernel) slot only, all other slots zero, and
                # read back the EE output slot -- this is exactly
                # pi * sum_theta[w * (...) * d2m2(theta, l)], up to a
                # constant prefactor from _corr2cl that cancels in the
                # Fl-normalized ratio below.
                corr_fl = np.zeros((2, 2, n))
                corr_fl[1, 1] = apod * csc2
                Fl = _corr2cl(corr_fl, (2, 2), lmax=key_lmax)[0, 0]

                corr_ee = np.zeros((2, 2, n))
                corr_ee[1, 1] = Xi_p + Xi_m
                corr_bb = np.zeros((2, 2, n))
                corr_bb[1, 1] = Xi_p - Xi_m
                # l < 2 is unphysical for spin-2 fields and always zero in
                # both Fl and the numerators (0/0); ignore the resulting
                # divide warning, the l < 2 entries are discarded below.
                with np.errstate(invalid="ignore"):
                    cl_EE = np.pi * _corr2cl(corr_ee, (2, 2), lmax=key_lmax)[0, 0] / Fl
                    cl_BB = np.pi * _corr2cl(corr_bb, (2, 2), lmax=key_lmax)[0, 0] / Fl

                # EB cross-term: unaffected by purification
                corr_eb = np.zeros((2, 2, n))
                corr_eb[0, 1] = wd[0, 1]
                corr_eb[1, 0] = wd[1, 0]
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
