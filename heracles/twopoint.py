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

from .core import TocDict, toc_match, update_metadata

logger = logging.getLogger(__name__)


def angular_power_spectra(
    alms,
    alms2=None,
    *,
    lmax=None,
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

        # minimum l for corrections
        lmin = max(map(abs, spins))

        # deconvolve the kernels of the first and second map
        for i, spin, kernel in zip([1, 2], spins, kernels):
            logger.info("- spin-%s %s kernel", spin, kernel)
            if kernel is None:
                fl = None
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
            else:
                msg = f"unknown kernel: {kernel}"
                raise ValueError(msg)
            if fl is not None:
                if cl.dtype.names is None:
                    cl[lmin:] /= fl[lmin:]
                else:
                    cl["CL"][lmin:] /= fl[lmin:]

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


def bin2pt(arr, bins, name, *, weights=None):
    """Compute binned two-point data."""

    def norm(a, b):
        """divide a by b if a is nonzero"""
        out = np.zeros(np.broadcast(a, b).shape)
        return np.divide(a, b, where=(a != 0), out=out)

    # flatten list of bins
    bins = np.reshape(bins, -1)
    m = bins.size

    # shape of the data
    n, *ds = np.shape(arr)
    ell = np.arange(n)

    # weights from string or given array
    if weights is None:
        w = np.ones(n)
    elif isinstance(weights, str):
        if weights == "l(l+1)":
            w = ell * (ell + 1)
        elif weights == "2l+1":
            w = 2 * ell + 1
        else:
            msg = f"unknown weights string: {weights}"
            raise ValueError(msg)
    else:
        w = np.asanyarray(weights)[:n]

    # create the structured output array
    # if input data is multi-dimensional, then so will the `name` column be
    binned = np.empty(
        m - 1,
        [
            ("L", float),
            (name, float, ds) if ds else (name, float),
            ("LMIN", float),
            ("LMAX", float),
            ("W", float),
        ],
    )

    # get the bin index for each ell
    i = np.digitize(ell, bins)

    assert i.size == ell.size

    # get the binned weights
    wb = np.bincount(i, weights=w, minlength=m)[1:m]

    # bin data in ell
    binned["L"] = norm(np.bincount(i, w * ell, m)[1:m], wb)
    for j in np.ndindex(*ds):
        x = (slice(None), *j)
        binned[name][x] = norm(np.bincount(i, w * arr[x], m)[1:m], wb)

    # add bin edges
    binned["LMIN"] = bins[:-1]
    binned["LMAX"] = bins[1:]

    # add weights
    binned["W"] = wb

    # all done
    return binned


def binned_cls(cls, bins, *, weights=None, out=None):
    """compute binned angular power spectra"""

    if out is None:
        out = TocDict()

    for key, cl in cls.items():
        out[key] = bin2pt(cl, bins, "CL", weights=weights)

    return out


def binned_mms(mms, bins, *, weights=None, out=None):
    """compute binned mixing matrices"""

    if out is None:
        out = TocDict()

    for key, mm in mms.items():
        out[key] = bin2pt(mm, bins, "MM", weights=weights)

    return out
