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
import numpy as np
import itertools
from copy import deepcopy
from itertools import combinations
from .utils import add_to_Cls, sub_to_Cls
from ..core import update_metadata
from ..result import Result, get_result_array
from ..mapping import transform
from ..twopoint import angular_power_spectra
from ..unmixing import _natural_unmixing, logistic
from ..transforms import cl2corr


def jackknife_cls(data_maps, vis_maps, jk_maps, fields, nd=1):
    """
    Compute the Cls of removing 1 Jackknife.
    inputs:
        data_maps (dict): Dictionary of data maps
        vis_maps (dict): Dictionary of visibility maps
        jkmaps (dict): Dictionary of mask maps
        fields (dict): Dictionary of fields
        nd (int): Number of Jackknife regions
    returns:
        cls (dict): Dictionary of data Cls
    """
    if nd < 0 or nd > 2:
        raise ValueError("number of deletions must be 0, 1, or 2")
    cls = {}
    mls0 = get_cls(vis_maps, jk_maps, fields)
    jkmap = jk_maps[list(jk_maps.keys())[0]]
    njk = len(np.unique(jkmap)[np.unique(jkmap) != 0])
    for regions in combinations(range(1, njk + 1), nd):
        _cls = get_cls(data_maps, jk_maps, fields, *regions)
        _cls_mm = get_cls(vis_maps, jk_maps, fields, *regions)
        # Mask correction
        alphas = mask_correction(_cls_mm, mls0)
        _cls = _natural_unmixing(_cls, alphas)
        # Bias correction
        _cls = correct_bias(_cls, jk_maps, fields, *regions)
        cls[regions] = _cls
    return cls


def get_cls(maps, jkmaps, fields, jk=0, jk2=0):
    """
    Internal method to compute the Cls of removing 2 Jackknife.
    inputs:
        maps (dict): Dictionary of data maps
        jkmaps (dict): Dictionary of mask maps
        fields (dict): Dictionary of fields
        jk (int): Jackknife region to remove
        jk2 (int): Jackknife region to remove
    returns:
        cls (dict): Dictionary of data Cls
    """
    print(f" - Computing Cls for regions ({jk},{jk2})", end="\r", flush=True)
    # remove the region from the maps
    _maps = jackknife_maps(maps, jkmaps, jk=jk, jk2=jk2)
    # compute alms
    alms = transform(fields, _maps)
    # compute cls
    cls = angular_power_spectra(alms)
    return cls


def jackknife_maps(maps, jkmaps, jk=0, jk2=0):
    """
    Internal method to remove a region from the maps.
    inputs:
        maps (dict): Dictionary of data maps
        jkmaps (dict): Dictionary of mask maps
        jk (int): Jackknife region to remove
        jk2 (int): Jackknife region to remove
    returns:
        maps (dict): Dictionary of data maps
    """
    _maps = deepcopy(maps)
    for key_data, key_mask in zip(maps.keys(), jkmaps.keys()):
        _map = _maps[key_data]
        _jkmap = jkmaps[key_mask]
        _mask = np.copy(_jkmap)
        _mask[_mask != 0] = _mask[_mask != 0] / _mask[_mask != 0]
        # Remove jk 2 regions
        cond = np.where((_jkmap == float(jk)) | (_jkmap == float(jk2)))[0]
        _mask[cond] = 0.0
        # Apply mask
        _map *= _mask
    return _maps


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
    returns:
        fskyjk2 (np.array): Fraction of the sky after deleting two regions.
    """
    rel_fskys = {}
    for key in jkmaps.keys():
        jkmap = jkmaps[key]
        mask = np.copy(jkmap)
        mask[mask != 0] = mask[mask != 0] / mask[mask != 0]
        fsky = sum(mask) / len(mask)
        cond = np.where((mask == 1.0) & (jkmap != jk) & (jkmap != jk2))[0]
        rel_fskys[key] = (len(cond) / len(mask)) / fsky
    return rel_fskys


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
        cls[key] = Result(cl)
    return cls


def mask_correction(Mljk, Mls0):
    """
    Internal method to compute the mask correction.
    input:
        Mljk (np.array): mask of delete1 Cls
        Mls0 (np.array): mask Cls
    returns:
        alpha (Float64): Mask correction factor
    """
    alphas = {}
    for key in list(Mljk.keys()):
        mljk = Mljk[key]
        mls0 = Mls0[key]
        # Transform to real space
        wmls0 = cl2corr(mls0)
        wmls0 = wmls0.T[0]
        wmljk = cl2corr(mljk)
        wmljk = wmljk.T[0]
        # Compute alpha
        alpha = wmljk / wmls0
        alpha *= logistic(np.log10(abs(wmljk)))
        alphas[key] = alpha
    return alphas


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
            result = Result(a, axis=axis, ell=ell)
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
        if i > 0:
            cov += (np.multiply.outer(delta, y - mu2) - cov) / i
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
            _qii = Result(_qii)
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
        Q[key] = Result(q_diag_exp, axis=q.axis, ell=q.ell)
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
        debiased_cov[key] = Result(
            c,
            ell=cov_jk[key].ell,
            axis=cov_jk[key].axis,
        )
    return debiased_cov
