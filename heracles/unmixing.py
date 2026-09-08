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


def _isolate(x, lmax):
    """
    Feed correlation function `x` through the spin-(2,2) "-" (Xi_m) slot and
    return the resulting E-mode (l-space) component. This is the building
    block of PolSpice's EE/BB "decouple" estimator (Chon et al. 2004, eq. 65):
    both the numerator (Xi_p +/- Xi_m of the masked data, weighted by the
    csc^2(theta/2) kernel) and the normalization Fl (the same kernel applied
    to the apodized mask correlation) are obtained by running the relevant
    theta-space quantity through this same transform.
    """
    n = x.shape[-1]
    corr = np.zeros((2, 2, n))
    corr[1, 1] = x
    return _corr2cl(corr, (2, 2), lmax=lmax)[0, 0]


def naturalspice(d, m, fields, theta_max=None, purify=False, apodization="logistic", progress: Progress | None = None):
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
        corr_wd = _naturalspice(wd, wm, fields, theta_max=theta_max, apodization=apodization, progress=task)

    # trnasform back to Cl
    if purify:
        with progress.task("purified transform back to Cl") as task:
            # start from the regular (natural/mask-ratio) transform, which
            # already gives us correct TT/TE/EB -- purification (PolSpice's
            # "decouple") only changes how EE/BB are estimated.
            corr_d = corr2cl(corr_wd)

            spin2_keys = [
                key for key, cwd in corr_wd.items() if cwd.spin[0] != 0 and cwd.spin[1] != 0
            ]
            current, total = 0, len(spin2_keys)
            for key in spin2_keys:
                current += 1
                task.update(current, total)

                # PolSpice's decoupled EE/BB (Chon et al. 2004, eq. 65;
                # deal_with_xi_and_cl.f90:do_cl_from_xi, decouple branch) is
                # built in two stages (spice_subs.f90: correct_xi_from_mask
                # then do_cl_from_xi): first the *natural* mask-ratio
                # correlation (Xi_p, Xi_m from corr_wd, same as TT/TE/EB
                # above), then a second Legendre transform of Xi_p +/- Xi_m
                # weighted by the csc^2(theta/2) kernel, normalized by Fl --
                # the same kernel applied to the apodization window alone.
                xvals = get_result_array(wd[key], "ell")[0]
                key_lmax = len(xvals) - 1
                theta = np.degrees(np.arccos(xvals))
                apod = gaussian_apod(theta, theta_max)
                with np.errstate(divide="ignore"):
                    csc2 = 1.0 / np.sin(np.radians(theta) / 2) ** 2

                Xi_p, Xi_m = corr_wd[key][0, 0], corr_wd[key][1, 1]

                fl = _isolate(apod * csc2, key_lmax)
                with np.errstate(invalid="ignore", divide="ignore"):
                    cl_EE = _isolate(Xi_p + Xi_m, key_lmax) / fl
                    cl_BB = _isolate(Xi_p - Xi_m, key_lmax) / fl

                cl = np.array(corr_d[key].array, copy=True)
                cl[0, 0] = cl_EE
                cl[1, 1] = cl_BB

                corr_d[key] = replace(corr_d[key], array=cl)
    else:
        with progress.task("transform back to Cl") as task:
            corr_d = corr2cl(corr_wd, progress=task)

    # truncate to lmax
    corr_d = binned(corr_d, np.arange(0, lmax + 1))
    return corr_d


def _naturalspice(wd, wm, fields, theta_max=None, apodization="logistic", progress: Progress | None = None):
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
        if apodization == "logistic":
            _wm *= logistic(np.log10(abs(_wm)), x0=x0)
        elif apodization == "gaussian":
            xvals = wm[m_key].ell
            theta = np.degrees(np.arccos(xvals))
            _wm /= gaussian_apod(theta, theta_max)
        corr_wds[key] = replace(wd[key], array=_wd/_wm)

    return corr_wds
