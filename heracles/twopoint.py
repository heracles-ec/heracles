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
"""module for angular power spectrum estimation"""

from __future__ import annotations

import logging
import time
from datetime import timedelta
from itertools import combinations_with_replacement, product
from typing import TYPE_CHECKING, Any

import numpy as np

from .core import TocDict, toc_match, update_metadata
from .progress import NoProgress, Progress
from .result import Result, binned

if TYPE_CHECKING:
    from collections.abc import Mapping, MutableMapping

    from numpy.typing import ArrayLike, NDArray

    from .fields import Field

# type alias for the keys of two-point data
TwoPointKey = tuple[Any, Any, Any, Any]

logger = logging.getLogger(__name__)


def alm2lmax(alm, mmax=None):
    """
    Returns the *lmax* value of the given *alm* array.
    """

    return (int((8 * np.shape(alm)[-1] + 1) ** 0.5 + 0.01) - 3) // 2


def alm2cl(alm, alm2=None, *, lmax=None):
    """
    Compute the angular power spectrum of spherical harmonic
    coefficients *alm*.  If *alm2* is given, return the angular
    cross-power spectrum of *alm* and *alm2*.  The spectrum is computed
    for all modes up to *lmax*, if given, or as many modes as provided
    in *alm* and *alm2* otherwise.
    """

    if alm2 is None:
        alm2 = alm

    alm = np.asanyarray(alm)
    alm2 = np.asanyarray(alm2)

    lmax1, lmax2 = alm2lmax(alm), alm2lmax(alm2)

    if lmax is None:
        lmax = step = min(lmax1, lmax2)
    else:
        step = min(lmax, lmax1, lmax2)

    if alm2.ndim > 1:
        alm = alm.reshape(*alm.shape[:-1], *((1,) * (alm2.ndim - 1)), alm.shape[-1])

    cl = alm.real[..., : step + 1] * alm2.real[..., : step + 1]

    start1 = lmax1 + 1
    start2 = lmax2 + 1
    for m in range(1, lmax + 1):
        stop1 = start1 + step - m + 1
        stop2 = start2 + step - m + 1
        a = alm.real[..., start1:stop1] * alm2.real[..., start2:stop2]
        b = alm.imag[..., start1:stop1] * alm2.imag[..., start2:stop2]
        cl[..., m:] += 2 * (a + b - cl[..., m:]) / (2 * m + 1)
        start1 += lmax1 - m + 1
        start2 += lmax2 - m + 1

    return cl


def _debias_cl(
    cl: NDArray[Any],
    bias: float | None = None,
    md: Mapping[str, Any] | None = None,
    *,
    inplace: bool = False,
) -> NDArray[Any]:
    """
    Remove additive bias from angular power spectrum.

    This function special-cases the bias from HEALPix maps.

    """

    if md is None:
        md = cl.dtype.metadata or {}

    if not inplace:
        cl = cl.copy()
        update_metadata(cl, **md)

    # use explicit bias, if given, or bias value from metadata
    if bias is None:
        bias = md.get("bias")
        # return early if there is no bias to be subtracted
        if bias is None:
            return cl

    # spins of the spectrum
    spin1, spin2 = md.get("spin_1", 0), md.get("spin_2", 0)

    # minimum and maximum angular mode for bias correction
    lmin = max(abs(spin1), abs(spin2))
    lmax = cl.shape[-1] - 1

    # this is what will be subtracted from the cl
    bl = np.zeros(cl.shape)
    if spin1 != 0 and spin2 != 0:
        # two spin-weighted fields: remove from EE and BB
        assert cl.shape[:2] == (2, 2)
        bl[[0, 1], [0, 1], ..., lmin:] = bias
    else:
        # other fields: remove from everywhere
        bl[..., lmin:] = bias

    # handle HEALPix pseudo-convolution
    for i, s in (1, spin1), (2, spin2):
        if md.get(f"kernel_{i}") == "healpix":
            import healpy as hp

            nside: int | None = md.get(f"nside_{i}")
            deconv: bool = md.get(f"deconv_{i}", True)
            if nside is not None and deconv:
                pw: NDArray[Any] | None
                if s == 0:
                    pw = hp.pixwin(nside, lmax=lmax, pol=False)
                elif s == 2:
                    _, pw = hp.pixwin(nside, lmax=lmax, pol=True)
                else:
                    pw = None
                if pw is not None:
                    bl[..., lmin:] /= pw[lmin:]

    # remove bias
    cl[:] -= bl

    return cl


def angular_power_spectra(
    alms,
    alms2=None,
    *,
    lmax=None,
    debias=True,
    bins=None,
    weights=None,
    include=None,
    exclude=None,
    out=None,
):
    """compute angular power spectra from a set of alms"""

    logger.info(
        "computing cls for %d%s alm(s)",
        len(alms),
        f"x{len(alms2)}" if alms2 is not None else "",
    )
    t = time.monotonic()

    logger.info("using LMAX = %s for cls", lmax)

    # collect all alm combinations for computing cls
    # iteration happens over keys only, values are accessed later
    if alms2 is None:
        pairs = combinations_with_replacement(alms, 2)
        alms2 = alms
    else:
        pairs = product(alms, alms2)

    # keep track of the twopoint combinations we have seen here
    twopoint_names = set()

    # output tocdict, use given or empty
    if out is None:
        cls = TocDict()
    else:
        cls = out

    # compute cls for all alm pairs
    # do not compute duplicates
    for (k1, i1), (k2, i2) in pairs:
        # skip duplicate cls in any order
        if (k1, k2, i1, i2) in cls or (k2, k1, i2, i1) in cls:
            continue

        # get the two-point code in standard order
        if (k1, k2) not in twopoint_names and (k2, k1) in twopoint_names:
            i1, i2 = i2, i1
            k1, k2 = k2, k1
            swapped = True
        else:
            swapped = False

        # check if cl is skipped by explicit include or exclude list
        if not toc_match((k1, k2, i1, i2), include, exclude):
            continue

        logger.info("computing %s x %s cl for bins %s, %s", k1, k2, i1, i2)

        # retrieve alms from keys; make sure swap is respected
        # this is done only now because alms might lazy-load from file
        if swapped:
            alm1, alm2 = alms2[k1, i1], alms[k2, i2]
        else:
            alm1, alm2 = alms[k1, i1], alms2[k2, i2]

        # compute the set of spectra from the pair of alms
        # this is stored as a block array
        cl = alm2cl(alm1, alm2, lmax=lmax)

        # build the metadata for the spectra from the alms
        # get metadata from alms
        md1 = alm1.dtype.metadata or {}
        md2 = alm2.dtype.metadata or {}
        # collect metadata
        md = {}
        bias = None
        for key, value in md1.items():
            if key == "bias":
                if k1 == k2 and i1 == i2:
                    bias = value
            else:
                md[f"{key}_1"] = value
        for key, value in md2.items():
            if key == "bias":
                pass
            else:
                md[f"{key}_2"] = value
        if bias is not None:
            md["bias"] = bias

        # debias cl if asked to
        if debias and bias is not None:
            _debias_cl(cl, bias, md, inplace=True)

        # write metadata for this spectrum
        update_metadata(cl, **md)

        # wrap in result array type
        # do this before binned() so it picks up the correct ell axes
        cl = Result(cl, axis=-1)

        # if bins are given, apply the binning
        if bins is not None:
            cl = binned(cl, bins, weights)

        # add cl to the set
        cls[k1, k2, i1, i2] = cl

        # keep track of names
        twopoint_names.add((k1, k2))

    logger.info(
        "computed %d cl(s) in %s",
        len(cls),
        timedelta(seconds=(time.monotonic() - t)),
    )

    # return the toc dict of cls
    return cls


def debias_cls(cls, bias=None, *, inplace=False):
    """remove bias from cls"""

    # the output toc dict
    out = cls if inplace else TocDict()

    # subtract bias of each cl in turn
    for key in cls:
        out[key] = _debias_cl(cls[key], bias and bias.get(key), inplace=inplace)

    # return the toc dict of debiased cls
    return out


def mixing_matrices(
    fields: Mapping[Any, Field],
    cls: Mapping[TwoPointKey, NDArray[Any]],
    *,
    l1max: int | None = None,
    l2max: int | None = None,
    l3max: int | None = None,
    bins: ArrayLike | None = None,
    weights: str | ArrayLike | None = None,
    out: MutableMapping[TwoPointKey, ArrayLike] | None = None,
    progress: Progress | None = None,
) -> MutableMapping[TwoPointKey, ArrayLike]:
    """compute mixing matrices for fields from a set of cls"""

    from convolvecl import mixmat, mixmat_eb

    # output dictionary if not provided
    if out is None:
        out = TocDict()

    # create dummy progress object if none was given
    if progress is None:
        progress = NoProgress()

    # inverse mapping of masks to fields
    masks: dict[str, dict[Any, Field]] = {}
    for key, field in fields.items():
        if field.mask is not None:
            if field.mask not in masks:
                masks[field.mask] = {}
            masks[field.mask][key] = field

    # keep track of combinations that have been done already
    done = set()

    # go through the toc dict of cls and compute mixing matrices
    # which mixing matrix is computed depends on the `masks` mapping
    current, total = 0, len(cls)
    for (k1, k2, i1, i2), cl in cls.items():
        current += 1
        progress.update(current, total)

        # if the masks are not named then skip this cl
        try:
            fields1 = masks[k1]
            fields2 = masks[k2]
        except KeyError:
            continue

        # deal with structured cl arrays
        if cl.dtype.names is not None:
            cl = cl["CL"]

        # compute mixing matrices for all fields of this mask combination
        for f1, f2 in product(fields1, fields2):
            # check if this combination has been done already
            if (f1, f2, i1, i2) in done or (f2, f1, i2, i1) in done:
                continue
            # otherwise, mark it as done
            done.add((f1, f2, i1, i2))

            with progress.task(f"({f1}, {f2}, {i1}, {i2})"):
                # get spins of fields
                spin1, spin2 = fields1[f1].spin, fields2[f2].spin

                # if any spin is zero, then there is no E/B decomposition
                if spin1 == 0 or spin2 == 0:
                    mixmat_or_mixmat_eb = mixmat
                else:
                    mixmat_or_mixmat_eb = mixmat_eb
                mm = mixmat_or_mixmat_eb(
                    cl,
                    l1max=l1max,
                    l2max=l2max,
                    l3max=l3max,
                    spin=(spin1, spin2),
                )

                # wrap in result array type
                # second to last axis is the *output* ell axis
                mm = Result(mm, axis=-2)

                if bins is not None:
                    mm = binned(mm, bins, weights)
                out[f1, f2, i1, i2] = mm
                del mm

    # return the toc dict of mixing matrices
    return out
