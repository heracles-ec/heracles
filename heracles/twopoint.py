# Heracles: Euclid code for harmonic-space statistics on the sphere
#
# Copyright (C) 2023 Euclid Science Ground Segment
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

import logging
import time
from datetime import timedelta
from itertools import combinations_with_replacement, product

import healpy as hp
import numpy as np
from convolvecl import mixmat, mixmat_eb

from .core import TocDict, toc_match
from .maps import (
    map_catalogs as _map_catalogs,
)
from .maps import (
    transform_maps as _transform_maps,
)
from .maps import (
    update_metadata,
)

logger = logging.getLogger(__name__)


def angular_power_spectra(alms, alms2=None, *, lmax=None, include=None, exclude=None):
    """compute angular power spectra from a set of alms"""

    logger.info(
        "computing cls for %d%s alm(s)",
        len(alms),
        f"x{len(alms2)}" if alms2 is not None else "",
    )
    t = time.monotonic()

    logger.info("using LMAX = %s for cls", lmax)

    # collect all alm combinations for computing cls
    if alms2 is None:
        alm_pairs = combinations_with_replacement(alms.items(), 2)
    else:
        alm_pairs = product(alms.items(), alms2.items())

    # keep track of the twopoint combinations we have seen here
    twopoint_names = set()

    # compute cls for all alm pairs
    # do not compute duplicates
    cls = TocDict()
    for ((k1, i1), alm1), ((k2, i2), alm2) in alm_pairs:
        # get the two-point code in standard order
        if (k1, k2) not in twopoint_names and (k2, k1) in twopoint_names:
            i1, i2 = i2, i1
            k1, k2 = k2, k1

        # skip duplicate cls in any order
        if (k1, k2, i1, i2) in cls or (k2, k1, i2, i1) in cls:
            continue

        # check if cl is skipped by explicit include or exclude list
        if not toc_match((k1, k2, i1, i2), include, exclude):
            continue

        logger.info("computing %s x %s cl for bins %s, %s", k1, k2, i1, i2)

        # compute the raw cl from the alms
        cl = hp.alm2cl(alm1, alm2, lmax_out=lmax)

        # collect metadata
        md = {}
        if alm1.dtype.metadata:
            for key, value in alm1.dtype.metadata.items():
                if key == "bias":
                    md[key] = value if k1 == k2 and i1 == i2 else 0.0
                else:
                    md[f"{key}_1"] = value
        if alm2.dtype.metadata:
            for key, value in alm2.dtype.metadata.items():
                if key == "bias":
                    pass
                else:
                    md[f"{key}_2"] = value
        update_metadata(cl, **md)

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

    logger.info("debiasing %d cl(s)%s", len(cls), " in place" if inplace else "")
    t = time.monotonic()

    # the output toc dict
    out = cls if inplace else TocDict()

    # subtract bias of each cl in turn
    for key in cls:
        logger.info("debiasing %s x %s cl for bins %s, %s", *key)

        cl = cls[key]
        md = cl.dtype.metadata or {}

        if not inplace:
            cl = cl.copy()
            update_metadata(cl, **md)

        # minimum l for correction
        lmin = max(abs(md.get("spin_1", 0)), abs(md.get("spin_2", 0)))

        # get bias from explicit dict, if given, or metadata
        if bias is None:
            b = md.get("bias", 0.0)
        else:
            b = bias.get(key, 0.0)

        # remove bias
        if cl.dtype.names is None:
            cl[lmin:] -= b
        else:
            cl["CL"][lmin:] -= b

        # write noise bias to corrected cl
        update_metadata(cl, bias=b)

        # store debiased cl in output set
        out[key] = cl

    logger.info(
        "debiased %d cl(s) in %s",
        len(out),
        timedelta(seconds=(time.monotonic() - t)),
    )

    # return the toc dict of debiased cls
    return out


def depixelate_cls(cls, *, inplace=False):
    """remove discretisation kernel from cls"""

    logger.info("depixelate %d cl(s)%s", len(cls), " in place" if inplace else "")
    t = time.monotonic()

    # keep a cache of convolution kernels (i.e. pixel window functions)
    fls = {
        "healpix": {},
    }

    # the output toc dict
    out = cls if inplace else TocDict()

    # remove effect of convolution for each cl in turn
    for key in cls:
        logger.info("depixelate %s x %s cl for bins %s, %s", *key)

        cl = cls[key]
        md = cl.dtype.metadata or {}

        if not inplace:
            cl = cl.copy()
            update_metadata(cl, **md)

        lmax = len(cl) - 1

        spins = [md.get("spin_1", 0), md.get("spin_2", 0)]
        kernels = [md.get("kernel_1"), md.get("kernel_2")]
        powers = [md.get("power_1", 0), md.get("power_2", 0)]
        areas = []

        # minimum l for corrections
        lmin = max(map(abs, spins))

        # deconvolve the kernels of the first and second map
        for i, spin, kernel in zip([1, 2], spins, kernels):
            logger.info("- spin-%s %s kernel", spin, kernel)
            if kernel is None:
                fl = None
                a = None
            elif kernel == "healpix":
                nside = md[f"nside_{i}"]
                if (nside, lmax, spin) not in fls[kernel]:
                    fl0, fl2 = hp.pixwin(nside, lmax=lmax, pol=True)
                    fls[kernel][nside, lmax, 0] = fl0
                    fls[kernel][nside, lmax, 2] = fl2
                fl = fls[kernel].get((nside, lmax, spin))
                if fl is None:
                    logger.warning(
                        "no HEALPix kernel for NSIDE = %s, LMAX = %s, SPIN = %s",
                        nside,
                        lmax,
                        spin,
                    )
                a = hp.nside2pixarea(nside)
            else:
                msg = f"unknown kernel: {kernel}"
                raise ValueError(msg)
            if fl is not None:
                if cl.dtype.names is None:
                    cl[lmin:] /= fl[lmin:]
                else:
                    cl["CL"][lmin:] /= fl[lmin:]
            areas.append(a)

        # scale by area**power
        for a, p in zip(areas, powers):
            if a is not None and p != 0:
                if cl.dtype.names is None:
                    cl[lmin:] /= a**p
                else:
                    cl["CL"][lmin:] /= a**p

        # store depixelated cl in output set
        out[key] = cl

    logger.info(
        "depixelated %d cl(s) in %s",
        len(out),
        timedelta(seconds=(time.monotonic() - t)),
    )

    # return the toc dict of depixelated cls
    return out


def mixing_matrices(cls, *, l1max=None, l2max=None, l3max=None):
    """compute mixing matrices from a set of cls"""

    logger.info("computing two-point mixing matrices for %d cl(s)", len(cls))
    t = time.monotonic()

    logger.info("using L1MAX = %s, L2MAX = %s, L3MAX = %s", l1max, l2max, l3max)

    # set of computed mixing matrices
    mms = TocDict()

    # go through the toc dict of cls and compute mixing matrices
    # which mixing matrix is computed depends on the combination of V/W maps
    for (k1, k2, i1, i2), cl in cls.items():
        if cl.dtype.names is not None:
            cl = cl["CL"]
        if k1 == "V" and k2 == "V":
            logger.info("computing 00 mixing matrix for bins %s, %s", i1, i2)
            w00 = mixmat(cl, l1max=l1max, l2max=l2max, l3max=l3max)
            mms["00", i1, i2] = w00
        elif k1 == "V" and k2 == "W":
            logger.info("computing 0+ mixing matrix for bins %s, %s", i1, i2)
            w0p = mixmat(cl, l1max=l1max, l2max=l2max, l3max=l3max, spin=(0, 2))
            mms["0+", i1, i2] = w0p
        elif k1 == "W" and k2 == "V":
            logger.info("computing 0+ mixing matrix for bins %s, %s", i2, i1)
            w0p = mixmat(cl, l1max=l1max, l2max=l2max, l3max=l3max, spin=(2, 0))
            mms["0+", i2, i1] = w0p
        elif k1 == "W" and k2 == "W":
            logger.info("computing ++, --, +- mixing matrices for bins %s, %s", i1, i2)
            wpp, wmm, wpm = mixmat_eb(
                cl,
                l1max=l1max,
                l2max=l2max,
                l3max=l3max,
                spin=(2, 2),
            )
            mms["++", i1, i2] = wpp
            mms["--", i1, i2] = wmm
            mms["+-", i1, i2] = wpm
        else:
            logger.warning(
                "computing unknown %s x %s mixing matrix for bins %s, %s",
                k1,
                k2,
                i1,
                i2,
            )
            w = mixmat(cl, l1max=l1max, l2max=l2max, l3max=l3max)
            mms[f"{k1}{k2}", i1, i2] = w

    logger.info(
        "computed %d mm(s) in %s",
        len(mms),
        timedelta(seconds=(time.monotonic() - t)),
    )

    # return the toc dict of mixing matrices
    return mms


def pixelate_mms_healpix(mms, nside, *, inplace=False):
    """apply HEALPix pixel window function to mms"""

    logger.info("pixelate %d mm(s)%s", len(mms), " in place" if inplace else "")
    logger.info("kernel: HEALPix, NSIDE=%d", nside)
    t = time.monotonic()

    # pixel window functions
    lmax = 4 * nside
    fl0, fl2 = hp.pixwin(nside, lmax=lmax, pol=True)

    # will be multiplied over rows
    fl0 = fl0[:, np.newaxis]
    fl2 = fl2[:, np.newaxis]

    # the output toc dict
    out = mms if inplace else TocDict()

    # apply discretisation kernel from cl to each mm in turn
    for key in mms:
        logger.info("pixelate %s mm for bins %s, %s", *key)

        mm = mms[key]
        if not inplace:
            mm = mm.copy()

        n = np.shape(mm)[-2]
        if n >= lmax:
            logger.error(
                "no HEALPix pixel window function for NSIDE=%d and LMAX=%d",
                nside,
                n - 1,
            )

        name = key[0]
        if name == "00":
            mm *= fl0[:n] * fl0[:n]
        elif name in ["0+", "+0"]:
            mm *= fl0[:n] * fl2[:n]
        elif name in ["++", "--", "+-", "-+"]:
            mm *= fl2[:n] * fl2[:n]
        else:
            logger.warning("unknown mixing matrix, assuming spin-0")
            mm *= fl0[:n] * fl0[:n]

        # store pixelated mm in output set
        out[key] = mm

    logger.info(
        "pixelated %d mm(s) in %s",
        len(out),
        timedelta(seconds=(time.monotonic() - t)),
    )

    # return the toc dict of modified cls
    return out


def binned_cls(cls, bins, *, weights=None, out=None):
    """compute binned angular power spectra"""

    def norm(a, b):
        """divide a by b if a is nonzero"""
        return np.divide(a, b, where=(a != 0), out=np.zeros_like(a))

    m = len(bins)

    if out is None:
        out = TocDict()

    for key, cl in cls.items():
        ell = np.arange(len(cl))

        if weights is None:
            w = np.ones_like(cl)
        elif isinstance(weights, str):
            if weights == "l(l+1)":
                w = ell * (ell + 1)
            elif weights == "2l+1":
                w = 2 * ell + 1
            else:
                msg = f"unknown weights string: {weights}"
                raise ValueError(msg)
        else:
            w = weights[: len(cl)]

        # create the output data
        binned = np.empty(
            m - 1,
            [
                ("L", float),
                ("CL", float),
                ("LMIN", float),
                ("LMAX", float),
                ("W", float),
            ],
        )

        # get the bin index for each ell
        i = np.digitize(ell, bins)

        # get the binned weights
        wb = np.bincount(i, weights=w, minlength=m)[1:m]

        # bin everything
        binned["L"] = norm(np.bincount(i, weights=w * ell, minlength=m)[1:m], wb)
        binned["CL"] = norm(np.bincount(i, weights=w * cl, minlength=m)[1:m], wb)

        # add bin edges
        binned["LMIN"] = bins[:-1]
        binned["LMAX"] = bins[1:]

        # add weights
        binned["W"] = wb

        # store result
        out[key] = binned

    return out


def random_bias(
    maps,
    catalogs,
    *,
    repeat=1,
    full=False,
    parallel=False,
    include=None,
    exclude=None,
    progress=False,
    **kwargs,
):
    """bias estimate from randomised maps

    The ``include`` and ``exclude`` selection is applied to the maps.

    """

    logger.info("estimating two-point bias for %d catalog(s)", len(catalogs))
    logger.info("randomising %s maps", ", ".join(map(str, maps)))
    t = time.monotonic()

    # grab lmax parameter if given
    lmax = kwargs.get("lmax", None)

    # include will be set below after we have the first set of alms
    include_cls = None
    if full:
        logger.info("estimating cross-biases")

    # set all input maps to randomize
    # store and later reset their initial state
    randomize = {k: m.randomize for k, m in maps.items()}
    try:
        for m in maps.values():
            m.randomize = True

        nbs = TocDict()

        for n in range(repeat):
            logger.info(
                "estimating bias from randomised maps%s",
                "" if n == 0 else f" (repeat {n+1})",
            )

            data = _map_catalogs(
                maps,
                catalogs,
                parallel=parallel,
                include=include,
                exclude=exclude,
                progress=progress,
            )
            alms = _transform_maps(data, progress=progress, **kwargs)

            # set the includes cls if full is false now that we know the alms
            if not full and include_cls is None:
                include_cls = [(k, k, i, i) for k, i in alms]

            cls = angular_power_spectra(alms, lmax=lmax, include=include_cls)

            for k, cl in cls.items():
                ell = np.arange(2, cl.shape[-1])
                nb = np.sum((2 * ell + 1) * cl[2:]) / np.sum(2 * ell + 1)
                nbs[k] = nbs.get(k, 0.0) + (nb - nbs.get(k, 0.0)) / (n + 1)
    finally:
        for k, m in maps.items():
            m.randomize = randomize[k]

    logger.info(
        "estimated %d two-point biases in %s",
        len(nbs),
        timedelta(seconds=(time.monotonic() - t)),
    )

    return nbs
