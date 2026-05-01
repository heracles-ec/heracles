# DICES: Euclid code for harmonic-space statistics on the sphere
#
# Copyright (C) 2023-2024 Euclid Science Ground Segment
#
# This file is part of DICES.
#
# DICES is free software: you can redistribute it and/or modify it
# under the terms of the GNU Lesser General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# DICES is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public
# License along with DICES. If not, see <https://www.gnu.org/licenses/>.
import os
import numpy as np
import itertools
from copy import deepcopy
from itertools import combinations
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
from ..utils import add_to_Cls, sub_to_Cls
from ..core import update_metadata
from ..result import Result, get_result_array, binned
from ..mapping import transform
from ..twopoint import angular_power_spectra
from ..unmixing import _naturalspice
from ..transforms import cl2corr, corr2cl
from ..io import write_alms, read_alms, write, read
from ..progress import NoProgress

try:
    from copy import replace
except ImportError:
    # Python < 3.13
    from dataclasses import replace


def _compute_cls_for_regions(args):
    """
    Worker function for one combination of deleted jackknife regions.
    Reads ALMs from the cache directory so no large arrays need to be pickled.
    """
    regions, dir, fields, jk_maps, mls0, mask_correction = args
    regions_tag = "_".join(map(str, regions))
    cls_path = os.path.join(dir, f"cls_{regions_tag}.fits")

    if os.path.exists(cls_path):
        return regions, read(cls_path)

    data_alms_full = read_alms(os.path.join(dir, "data_alms_0.fits"))
    alms_jk = _subtract_alms(
        data_alms_full,
        _accumulate_alms(os.path.join(dir, f"data_alms_{r}.fits") for r in regions),
    )
    _cls = angular_power_spectra(alms_jk)
    _cls = correct_bias(_cls, jk_maps, fields, *regions)

    if mask_correction == "Full":
        vis_alms_full = read_alms(os.path.join(dir, "vis_alms_0.fits"))
        vis_alms_jk = _subtract_alms(
            vis_alms_full,
            _accumulate_alms(os.path.join(dir, f"vis_alms_{r}.fits") for r in regions),
        )
        _cls_mm = angular_power_spectra(vis_alms_jk)
        _cls = correct_footprint_naturalspice(_cls, _cls_mm, mls0, fields)
    elif mask_correction == "Fast":
        _cls = correct_footprint_fsky(_cls, jk_maps, fields, *regions)
    else:
        raise ValueError("mask_correction must be 'Fast' or 'Full'")

    write(cls_path, _cls, clobber=True)
    return regions, _cls


def jackknife_cls(
    data_maps,
    vis_maps,
    jk_maps,
    fields,
    mask_correction="Fast",
    nd=1,
    dir="./dices",
    parallel=False,
    progress=None,
):
    """
    Compute the Cls of removing 1 Jackknife.
    inputs:
        data_maps (dict): Dictionary of data maps
        vis_maps (dict): Dictionary of visibility maps
        jk_maps (dict): Dictionary of mask maps
        fields (dict): Dictionary of fields
        mask_correction (str): Type of mask correction to apply ("Fast" or "Full")
        nd (int): Number of Jackknife regions
        dir (str): Directory for caching intermediate ALMs.
        parallel (bool): If True, compute Cls in parallel using all available cores minus one.
        progress (Progress): Progress reporter.
    returns:
        cls (dict): Dictionary of data Cls
    """
    if nd < 0 or nd > 2:
        raise ValueError("number of deletions must be 0, 1, or 2")

    if progress is None:
        progress = NoProgress()

    cls = {}
    jkmap = jk_maps[list(jk_maps.keys())[0]]
    njk = len(np.unique(jkmap)[np.unique(jkmap) != 0])
    os.makedirs(dir, exist_ok=True)

    # Compute ALMs
    progress.update(0, njk + 1)
    for k in range(0, njk + 1):
        data_path = os.path.join(dir, f"data_alms_{k}.fits")
        vis_path = os.path.join(dir, f"vis_alms_{k}.fits")
        with progress.task(f"ALMs {k}"):
            if not (os.path.exists(data_path) and os.path.exists(vis_path)):
                if k == 0:
                    data_alms_k = transform(fields, data_maps)
                    vis_alms_k = transform(fields, vis_maps)
                else:
                    data_alms_k = transform(
                        fields, _get_region_maps(data_maps, jk_maps, k)
                    )
                    vis_alms_k = transform(
                        fields, _get_region_maps(vis_maps, jk_maps, k)
                    )
                write_alms(data_path, data_alms_k, clobber=True)
                write_alms(vis_path, vis_alms_k, clobber=True)
        progress.update(k + 1, njk + 1)

    # Compute Cls
    vis_alms_full = read_alms(os.path.join(dir, "vis_alms_0.fits"))
    mls0 = angular_power_spectra(vis_alms_full)

    all_regions = list(combinations(range(1, njk + 1), nd))
    args_list = [
        (regions, dir, fields, jk_maps, mls0, mask_correction)
        for regions in all_regions
    ]

    n_regions = len(all_regions)
    progress.update(0, n_regions)
    if not parallel:
        for i, args in enumerate(args_list):
            with progress.task(f"Cls regions {args[0]}"):
                regions, _cls = _compute_cls_for_regions(args)
            cls[regions] = _cls
            progress.update(i + 1, n_regions)
    else:
        nworkers = max(1, (os.cpu_count() or 1) - 1)
        with ProcessPoolExecutor(
            max_workers=nworkers, mp_context=get_context("spawn")
        ) as executor:
            futures = {
                executor.submit(_compute_cls_for_regions, args): args[0]
                for args in args_list
            }
            for i, future in enumerate(as_completed(futures)):
                regions, _cls = future.result()
                cls[regions] = _cls
                progress.update(i + 1, n_regions)

    return cls


def _get_region_maps(maps, jkmaps, jk):
    """
    Returns maps with only the pixels belonging to jackknife region *jk* active.
    All other pixels are set to zero.
    """
    _maps = deepcopy(maps)
    for key_data, key_mask in zip(maps.keys(), jkmaps.keys()):
        _map = _maps[key_data]
        _jkmap = jkmaps[key_mask]
        if _jkmap is None:
            continue
        _mask = (_jkmap == float(jk)).astype(int)
        _map *= _mask
    return _maps


def _sum_alms_except(alms_regions, exclude=()):
    """
    Returns the sum of all region alms except those whose key is in *exclude*.

    Metadata (including bias) is taken from the first included region's alms,
    consistent with the mapper copying the full-footprint bias to every region.
    Passing an empty *exclude* gives the full-sky alms; passing the deleted
    region keys gives the delete-k or delete-k1k2 alms directly.
    """
    included = [alms for k, alms in alms_regions.items() if k not in exclude]
    first = included[0]
    result = {}
    for key in first:
        arr = first[key].copy()
        for alms_k in included[1:]:
            arr += alms_k[key]
        result[key] = arr
    return result


def _accumulate_alms(paths):
    """Reads ALMs from each path and returns their sum, loading one file at a time."""
    result = None
    for path in paths:
        alms = read_alms(path)
        if result is None:
            result = {key: arr.copy() for key, arr in alms.items()}
        else:
            for key in result:
                result[key] += alms[key]
    return result


def _subtract_alms(full_alms, region_sum):
    """Returns full_alms minus region_sum, or a copy of full_alms if region_sum is None."""
    result = {}
    for key in full_alms:
        result[key] = full_alms[key].copy()
        if region_sum is not None:
            result[key] -= region_sum[key]
    return result


def bias(cls):
    """
    Internal method to compute the bias.
    inputs:
        cls (dict): Dictionary of Cls
    returns:
        bias (dict): Dictionary
    """
    bias = {}
    for key in list(cls.keys()):
        meta = cls[key].dtype.metadata
        bias[key] = meta.get("bias", 0)
    return bias


def jackknife_fsky(jkmaps, jk=0, jk2=0):
    """
    Returns the fraction of the sky after deleting two regions.
    inputs:
        jkmaps (dict): Dictionary of Jackknife maps
        jk (int): Jackknife region to remove
        jk2 (int): Jackknife region to remove
        ratio (bool): Return the ratio of fskyjk to fsky
    returns:
        fskyjk2 (np.array): Fraction of the sky after deleting two regions.
    """
    fskysjk = {}
    for key in jkmaps.keys():
        jkmap = jkmaps[key]
        mask = np.copy(jkmap)
        mask = (mask > 0).astype(int)
        fsky = sum(mask) / len(mask)
        cond = np.where((mask == 1.0) & (jkmap != jk) & (jkmap != jk2))[0]
        fskyjk = len(cond) / len(mask)
        fskysjk[key] = fskyjk / fsky
    return fskysjk


def jackknife_bias(bias, fsky, fields):
    """
    Returns the bias for deleting a Jackknife region.
    inputs:
        bias (dict): Dictionary of biases
        fsky (dict): Dictionary of relative fskys
        fields (dict): Dictionary of fields
    returns:
        bias_jk (dict): Dictionary of biases
    """
    bias_jk = {}
    for key in list(bias.keys()):
        f1, f2, b1, b2 = key
        b = bias[key]
        if (f1, b1) == (f2, b2):
            field = fields[f1]
            m_f = field.mask
            fskyjk = fsky[(m_f, b1)]
        else:
            fskyjk = 0.0
        b_jk = b * fskyjk
        bias_jk[key] = b_jk
    return bias


def correct_bias(cls, jkmaps, fields, jk=0, jk2=0):
    """
    Corrects the bias of the Cls due to taking out a region.
    inputs:
        cls (dict): Dictionary of Cls
        jkmaps (dict): Dictionary of Jackknife maps
        fields (dict): Dictionary of fields
        jk (int): Jackknife region to remove
        jk2 (int): Jackknife region to remove
    returns:
        cls_cbias (dict): Corrected Cls
        cls_wbias (dict): Cls with bias
    """
    # Bias correction
    b = bias(cls)
    fskyjk = jackknife_fsky(jkmaps, jk=jk, jk2=jk2)
    b_jk = jackknife_bias(b, fskyjk, fields)
    # Correct Cls
    cls = add_to_Cls(cls, b)
    cls = sub_to_Cls(cls, b_jk)
    # Update metadata
    for key in cls.keys():
        cl = cls[key].array
        update_metadata(cl, bias=b_jk[key])
        cls[key] = replace(cls[key], array=cl)
    return cls


def correct_footprint_fsky(cls, jkmaps, fields, jk=0, jk2=0):
    """
    Corrects the Cls for the footprint reduction due to taking out a region.
    inputs:
        cls (dict): Dictionary of Cls
        jkmaps (dict): Dictionary of Jackknife maps
        fields (dict): Dictionary of fields
        jk (int): Jackknife region to remove
        jk2 (int): Jackknife region to remove
    returns:
        cls_cf (dict): Corrected Cls
    """
    fskyjk = jackknife_fsky(jkmaps, jk=jk, jk2=jk2)
    _cls = {}
    for key in cls.keys():
        a, b, i, j = key
        f_a = fields[a]
        f_b = fields[b]
        m_a = f_a.mask
        m_b = f_b.mask
        fsky_a = fskyjk[(m_a, i)]
        fsky_b = fskyjk[(m_b, j)]
        _cl = cls[key].array / np.sqrt(fsky_a * fsky_b)
        _cls[key] = replace(cls[key], array=_cl)
    return _cls


def _mask_correlation_ratio(mljk, mls0):
    alphas = {}
    wmls0 = cl2corr(mls0)
    wmljk = cl2corr(mljk)
    for key in list(wmljk.keys()):
        _wmljk = wmljk[key].array
        _wmls0 = wmls0[key].array
        alpha = _wmljk / _wmls0
        alphas[key] = replace(mls0[key], array=alpha)
    return alphas


def correct_footprint_naturalspice(cls, cls_mm, mls0, fields):
    """
    Corrects the Cls for footprint reduction using the full NaMaster/naturalspice approach.
    inputs:
        cls (dict): Dictionary of data Cls
        cls_mm (dict): Dictionary of jackknife mask Cls
        mls0 (dict): Dictionary of full mask Cls
        fields (dict): Dictionary of fields
    returns:
        cls (dict): Corrected Cls
    """
    alphas = _mask_correlation_ratio(cls_mm, mls0)
    first_cls = list(cls.values())[0]
    first_mls = list(mls0.values())[0]
    lmax = first_cls.shape[first_cls.axis[0]]
    lmax_mask = first_mls.shape[first_mls.axis[0]]
    cls = binned(cls, np.arange(0, lmax_mask + 1))
    wcls = cl2corr(cls)
    wcls = _naturalspice(wcls, alphas, fields)
    cls = corr2cl(wcls)
    return binned(cls, np.arange(0, lmax + 1))


def jackknife_covariance(dict, nd=1):
    """
    Compute the jackknife covariance matrix from a sequence
    of spectra dictionaries *dict*.
    """
    return _jackknife_covariance(dict.values(), nd=nd)


def _jackknife_covariance(samples, nd=1):
    """
    Compute the jackknife covariance matrix from a sequence
    of spectra dictionaries *samples*.
    """
    cov = {}
    # first sample is the blueprint that rest must follow
    first, *rest = samples
    # loop over pairs of keys to compute their covariance
    for key1, key2 in itertools.combinations_with_replacement(first, 2):
        # get reference results
        result1 = first[key1]
        result2 = first[key2]
        sa1, sb1 = result1.spin
        sa2, sb2 = result2.spin
        # gather samples for this key combination
        samples1 = np.stack([result1] + [spectra[key1] for spectra in rest])
        samples2 = np.stack([result2] + [spectra[key2] for spectra in rest])
        # if there are multiple samples, compute covariance
        if (m := len(samples1)) > 1:
            # compute jackknife covariance matrix
            a = sample_covariance(samples1, samples2)
            if nd == 1:
                njk = m
                a *= (njk - 1) ** 2 / njk
            elif nd == 2:
                njk = (1 + np.sqrt(1 + 8 * m)) / 2
                a *= (njk * (njk - 1) - 2) / (2 * njk * (njk + 1))
            elif nd > 2:
                raise ValueError("number of deletions must be 0, 1, or 2")
            # move ell axes last, in order
            ndim1 = result1.ndim
            oldaxis = result1.axis + tuple(ndim1 + ax for ax in result2.axis)
            axis = tuple(range(-len(oldaxis), 0))
            a = np.moveaxis(a, oldaxis, axis)
            # get attributes of result
            ell = get_result_array(result1, "ell")
            ell += get_result_array(result2, "ell")
            # add extra axis if needed
            a1, b1, i1, j1 = key1
            a2, b2, i2, j2 = key2
            result = Result(a, axis=axis, spin=(sa1, sb1, sa2, sb2), ell=ell)
            # store result
            cov[a1, b1, a2, b2, i1, j1, i2, j2] = result
    return cov


def sample_covariance(samples, samples2=None):
    """
    Returns the sample covariance matrix of *samples*, or the sample
    cross-covariance between *samples* and *samples2* if the latter is
    given.
    """
    if samples2 is None:
        samples2 = samples
    n, *dim = samples.shape
    n2, *dim2 = samples2.shape
    if n2 != n:
        raise ValueError("different numbers of samples")
    mu = np.zeros((*dim,))
    mu2 = np.zeros((*dim2,))
    cov = np.zeros((*dim, *dim2))
    for i in range(n):
        x = samples[i]
        y = samples2[i]
        delta = x - mu
        mu += delta / (i + 1)
        mu2 += (y - mu2) / (i + 1)
        cov += (np.multiply.outer(delta, y - mu2) - cov) / (i + 1)
    # Renormalize covariance
    cov *= n / (n - 1)
    return cov


def delete2_correction(cls0, cls1, cls2):
    """
    Internal method to compute the delete2 correction.
    inputs:
        Cls0 (dict): Dictionary of data Cls
        Cls1 (dict): Dictionary of delete1 data Cls
        Cls2 (dict): Dictionary of delete2 data Cls
    returns:
        Q (dict): Dictionary of delete2 correction
    """
    # Compute the ensemble for the correction
    Q_ii = []
    Njk = len(cls1)
    for kk in cls2:
        k1, k2 = kk
        qii = {}
        for key in cls2[kk]:
            _qii = Njk * cls0[key].array
            _qii -= (Njk - 1) * cls1[(k1,)][key].array
            _qii -= (Njk - 1) * cls1[(k2,)][key].array
            _qii += (Njk - 2) * cls2[kk][key].array
            _qii = replace(cls0[key], array=_qii)
            qii[key] = _qii
        Q_ii.append(qii)
    # Compute the correction from the ensemble
    Q = _jackknife_covariance(Q_ii, nd=2)
    # Diagonalise the correction
    for key in Q:
        q = Q[key]
        *_, length = q.shape
        q_diag = np.diagonal(q, axis1=-2, axis2=-1)
        q_diag_exp = np.zeros_like(q)
        diag_indices = np.arange(length)  # Indices for the diagonal
        q_diag_exp[..., diag_indices, diag_indices] = q_diag
        Q[key] = replace(q, array=q_diag_exp)
    return Q


def debias_covariance(cov_jk, cls0, cls1, cls2):
    """
    Debiases the Jackknife covariance using the delete2 ensemble.
    inputs:
        cov_jk (dict): Dictionary of delete1 covariance
        cls0 (dict): Dictionary of data Cls
        cls1 (dict): Dictionary of delete1 data Cls
        cls2 (dict): Dictionary of delete2 data Cls
    returns:
        debiased_cov (dict): Dictionary of debiased Jackknife covariance
    """
    Q = delete2_correction(cls0, cls1, cls2)
    return _debias_covariance(cov_jk, Q)


def _debias_covariance(cov_jk, Q):
    """
    Internal method to debias the Jackknife covariance.
    Useful when the delete2 correction is already computed.
    """
    debiased_cov = {}
    for key in list(cov_jk.keys()):
        c = cov_jk[key].array - Q[key].array
        debiased_cov[key] = replace(cov_jk[key], array=c)
    return debiased_cov
