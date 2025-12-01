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


def tune_direct_inversion(data_cls, mms, target_cls, cov, maxiter=10):
    """
    Tune the natural unmixing parameters to minimize the difference between
    the unmixing-corrected data Cl and the target Cl.
    Args:
        data_cls: Data Cl
        mls: mask Cl
        target_cls: Target Cl
        cov: Covariance of the data Cl
        fields: list of fields
    Returns:
        tuned_params: Dictionary with the tuned parameters
    """
    from scipy.optimize import minimize_scalar

    options = {}
    for key, data_cl in data_cls.items():
        print(f"Tuning natural unmixing for key: {key}")
        # create a dictionary only with the current key
        a, b, i, j = key
        cov_key = (a, b, a, b, i, j, i, j)
        # The objective function to minimize depends
        # on the spin of the wcl
        s1, s2 = data_cl.spin
        if s1 == 0 and s2 == 0:
            _data_cl = data_cl.array
            _target_cl = target_cls[key].array
            mm = get_cl(key, mms).array
            U, s, Vt = np.linalg.svd(mm, full_matrices=False)
            inv_cov = np.linalg.pinv(cov[cov_key].array)
        if (s1 != 0 and s2 == 0) or (s1 == 0 and s2 != 0):
            _data_cl = data_cl.array[0, :]
            _target_cl = target_cls[key].array[0, :]
            mm = get_cl(key, mms).array
            U, s, Vt = np.linalg.svd(mm, full_matrices=False)
            inv_cov = np.linalg.pinv(cov[cov_key].array[0, 0, :, :])
        if s1 != 0 and s2 != 0:
            _data_cl = data_cl.array[0, 0, :]
            _target_cl = target_cls[key].array[0, 0, :]
            mm = get_cl(key, mms)[0, :, :]
            print("mm shape: ", get_cl(key, mms).shape)
            U, s, Vt = np.linalg.svd(mm, full_matrices=False)
            inv_cov = np.linalg.pinv(cov[cov_key].array[0, 0, 0, 0, :, :])

        def objective(rtol):
            # Invert singular values with cutoff
            cutoff = rtol * np.max(s)
            s_inv = np.array([1 / si if si > cutoff else 0 for si in s])
            inv_mm = (Vt.T * s_inv) @ U.T
            corr_cl = inv_mm @ _data_cl
            diff = corr_cl - _target_cl
            xi2 = diff.T @ inv_cov @ diff
            return xi2

        opt_xi2 = minimize_scalar(
            objective, bounds=(0.2, 1), method="bounded", options={"maxiter": maxiter}
        )
        options[key] = opt_xi2.x
    return options


def tune_natural_unmixing(data_cls, mls, target_cls, cov, fields, maxiter=10):
    """
    Tune the natural unmixing parameters to minimize the difference between
    the unmixing-corrected data Cl and the target Cl.
    Args:
        data_cls: Data Cl
        mls: mask Cl
        target_cls: Target Cl
        cov: Covariance of the data Cl
        fields: list of fields
    Returns:
        tuned_params: Dictionary with the tuned parameters
    """
    from scipy.optimize import minimize_scalar

    mask_lmax = mls[list(mls.keys())[0]].shape[-1]
    wmls = transform_cls(mls)
    data_wcls = transform_cls(data_cls, lmax_out=mask_lmax)

    options = {}
    inv_covs = {}
    for key, data_wcl in data_wcls.items():
        print(f"Tuning natural unmixing for key: {key}")
        # create a dictionary only with the current key
        a, b, i, j = key
        cov_key = (a, b, a, b, i, j, i, j)
        _data_wcls = {key: data_wcl}
        _target_cls = {key: target_cls[key]}
        # The objective function to minimize depends
        # on the spin of the wcl
        s1, s2 = data_wcl.spin

        def objective(rtol):
            corr_wmls = correct_correlation(wmls, rtol=rtol)
            corr_cls = _natural_unmixing(_data_wcls, corr_wmls, fields)
            corr_cl = get_cl(key, corr_cls).array
            target_cl = get_cl(key, _target_cls).array
            if s1 == 0 and s2 == 0:
                diff = corr_cl - target_cl
                if cov_key not in inv_covs:
                    inv_covs[cov_key] = np.linalg.pinv(cov[cov_key].array)
            if (s1 != 0 and s2 == 0) or (s1 == 0 and s2 != 0):
                diff = corr_cl[0, :] - target_cl[0, :]
                if cov_key not in inv_covs:
                    inv_covs[cov_key] = np.linalg.pinv(cov[cov_key].array[0, 0, :, :])
            if s1 != 0 and s2 != 0:
                diff = corr_cl[0, 0, :] - target_cl[0, 0, :]
                if cov_key not in inv_covs:
                    inv_covs[cov_key] = np.linalg.pinv(
                        cov[cov_key].array[0, 0, 0, 0, :, :]
                    )
            inv_cov = inv_covs[cov_key]
            xi2 = diff.T @ inv_cov @ diff
            return xi2

        opt_xi2 = minimize_scalar(
            objective, bounds=(0.2, 1), method="bounded", options={"maxiter": maxiter}
        )
        options[key] = opt_xi2.x
    return options


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
    lmax = cls[list(cls.keys())[0]].shape[-1]
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
        wcl = wcls[key].array
        wcl /= wml
        corr_wcls[key] = replace(wcls[key], array=wcl)

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
        wm *= logistic(np.log10(abs(wm)), x0=np.log10(cutoff))
        corrected_wms[key] = replace(wms[key], array=wm)
    return corrected_wms


def logistic(x, x0=-5, k=50):
    return 1.0 + np.exp(-k * (x - x0))
