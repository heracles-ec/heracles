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
from ..result import (
    get_result_array,
    Result,
)


def jackknife_covariance(samples, nd=1):
    """
    Compute the jackknife covariance matrix from a sequence
    of spectra dictionaries *samples*.
    """
    cov = {}
    # no samples means no covariance
    if not samples:
        return cov
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
        if (njk := len(samples1)) > 1:
            # compute jackknife covariance matrix
            a = sample_covariance(samples1, samples2)
            if nd == 1:
                a *= (njk - 1)
            elif nd == 2:
                a *= (njk * (njk - 1) - 2) / (2 * njk * (njk + 1))
            elif nd > 2:
                raise ValueError("number of deletions must be 0, 1, or 2")
            # move ell axes last, in order
            ndim1 = result1.ndim
            oldaxis = result1.axis + tuple(ndim1 + ax for ax in result2.axis)
            axis = tuple(range(-len(oldaxis), 0))
            a = np.moveaxis(a, oldaxis, axis)
            # get attributes of result
            ell = (
                get_result_array(result1, "ell"),
                get_result_array(result2, "ell"),
            )
            # wrap everything into a result instance
            result = Result(a, axis=axis, ell=ell)
            # store result
            a1, b1, i1, j1 = key1
            a2, b2, i2, j2 = key2
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
    return cov
