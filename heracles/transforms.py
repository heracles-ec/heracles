import numpy as np


def legendre_p_all(n, x, *, diff_n=0):
    """
    All Legendre polynomials of the first kind up to the specified degree n,
    evaluated at a scalar x, optionally with their first derivatives.
    """
    allP = np.empty(n + 1)
    allP[0] = 1.0
    if n >= 1:
        allP[1] = x
    for l in range(1, n):
        allP[l + 1] = ((2 * l + 1) * x * allP[l] - l * allP[l - 1]) / (l + 1)
    if diff_n == 0:
        return allP
    assert diff_n == 1, "only diff_n=1 is supported"
    ls = np.arange(1, n + 1)
    alldP = np.zeros_like(allP)
    alldP[1:] = ls * (allP[:-1] - x * allP[1:]) / (1 - x**2)
    return allP, alldP


try:
    from copy import replace
except ImportError:
    # Python < 3.13
    from dataclasses import replace

from .progress import NoProgress, Progress
from .result import get_result_array

gauss_legendre = None
_gauss_legendre_cache = {}


def _cached_gauss_legendre(npoints, cache=True):
    """
    Gauss-Legendre quadrature nodes/weights for `npoints` points on [-1, 1].
    """
    key = npoints
    if cache and key in _gauss_legendre_cache:
        return _gauss_legendre_cache[key]
    else:
        if gauss_legendre is not None:
            xvals = np.empty(npoints)
            weights = np.empty(npoints)
            gauss_legendre(xvals, weights, npoints)
            xvals.flags.writeable = False
            weights.flags.writeable = False
        else:
            xvals, weights = np.polynomial.legendre.leggauss(npoints)
            xvals.flags.writeable = False
            weights.flags.writeable = False
        if cache:
            _gauss_legendre_cache[key] = xvals, weights
        return xvals, weights


def legendre_funcs(lmax, x, spin, lfacs=None, lfacs2=None, lrootfacs=None):
    """
    Utility function to return the Legendre/Wigner-d functions needed to
    transform a Cl of the given `spin` to/from a correlation function, for
    all :math:`\ell` up to lmax. Note that the spin functions start at
    :math:`\ell=2`, so are shorter than the spin (0, 0) case.

    Only spin (0, 0), (0, 2)/(2, 0), and (2, 2) are supported.

    :param lmax: maximum :math:`\ell`
    :param x: scalar value of :math:`\cos(\theta)` at which to evaluate
    :param spin: (s1, s2) spin of the field pair -- selects which functions
        are computed
    :param lfacs: optional pre-computed :math:`\ell(\ell+1)` float array
        (ignored for spin (0, 0))
    :param lfacs2: optional pre-computed :math:`(\ell+2)*(\ell-1)` float array
    :param lrootfacs: optional pre-computed sqrt(lfacs*lfacs2) array
    :return: `P`, starting at :math:`\ell=0`, for spin (0, 0); otherwise
        `(d_{20}, d_{22}, d_{2,-2})`, starting at :math:`\ell=2`
    """
    s1, s2 = spin
    if s1 == 0 and s2 == 0:
        return legendre_p_all(lmax, x)
    elif not ({s1, s2} == {0, 2} or (s1 == 2 and s2 == 2)):
        raise ValueError(
            f"unsupported spin combination {spin!r}: only (0, 0), "
            "(0, 2)/(2, 0), and (2, 2) are supported"
        )

    allP, alldP = legendre_p_all(lmax, x, diff_n=1)
    fac1 = 1 - x
    fac2 = 1 + x

    if lfacs is None:
        ls = np.arange(2, lmax + 1, dtype=np.float64)
        lfacs = ls * (ls + 1)
        lfacs2 = (ls + 2) * (ls - 1)
        lrootfacs = np.sqrt(lfacs * lfacs2)
    P = allP[2:]
    dP = alldP[2:]

    fac = fac1 / fac2
    d22 = (
        ((4 * x - 8) / fac2 + lfacs) * P + 4 * fac * (fac2 + (x - 2) / lfacs) * dP
    ) / lfacs2
    if x > 0.998:
        # for stability use series at small angles (thanks Pavel Motloch)
        d2m2 = np.empty(lmax - 1)
        indser = int(np.sqrt((400.0 + 3 / (1 - x**2)) / 150)) - 1
        d2m2[indser:] = (
            (lfacs[indser:] - (4 * x + 8) / fac1) * P[indser:]
            + 4 / fac * (-fac1 + (x + 2) / lfacs[indser:]) * dP[indser:]
        ) / lfacs2[indser:]
        sin2 = 1 - x**2
        d2m2[:indser] = (
            lfacs[:indser]
            * lfacs2[:indser]
            * sin2**2
            / 7680
            * (20 + sin2 * (16 - lfacs[:indser]))
        )
    else:
        d2m2 = (
            (lfacs - (4 * x + 8) / fac1) * P + 4 / fac * (-fac1 + (x + 2) / lfacs) * dP
        ) / lfacs2
    d20 = (2 * x * dP - lfacs * P) / lrootfacs
    return d20, d22, d2m2


def _cl2corr(cl, kernel, lmax=None, sampling_factor=1, xvals=None):
    """
    Get the correlation function of a single, already-rotated Cl COMPONENT
    (1D array `cl[l]`) from the power spectrum, evaluated at points
    cos(theta) = xvals, via the explicit kernel selected by `kernel`:
    (0, 0) -> Legendre P_l (l from 0); (2, 2) -> Wigner d^l_{2,2}; (2, -2)
    -> d^l_{2,-2}; (2, 0) (alias (0, 2)) -> d^l_{2,0} (l from 2 for the
    polarization kernels). `cl` must already be the desired linear
    combination of physical fields (e.g. Cl^EE+Cl^BB for kernel (2, 2)) --
    add/subtract the relevant pair first to build that combination from a
    full [[EE, EB], [BE, BB]] (or [Ta, Tb]) array.
    Use roots of Legendre polynomials (np.polynomial.legendre.leggauss) for accurate back integration with corr2cl.
    Note currently does not work at xvals=1 (can easily calculate that as special case!).

    :param cl: Cl component array, `cl[l]`. Should include
        :math:`\ell(\ell+1)/2\pi` factors.
    :param kernel: (s1, s2) kernel selector; only (0, 0), (2, 0)/(0, 2),
        (2, 2), and (2, -2) are supported
    :param lmax: optional maximum L to use from the cl array
    :param sampling_factor: oversampling factor for the quadrature grid,
        ignored if `xvals` is given
    :param xvals: if given, evaluate the correlation function at these
        cos(theta) points directly instead of the Gauss-Legendre quadrature
        grid -- e.g. to evaluate at arbitrary angles, not just quadrature
        nodes (`_corr2cl` then cannot be used to transform the result back)
    :return: correlation function component array, the l axis replaced by
        the quadrature (theta) axis
    """
    cl = np.asarray(cl, dtype=np.float64)

    if lmax is None:
        lmax = cl.shape[-1] - 1

    if xvals is None:
        xvals, _ = _cached_gauss_legendre(int(sampling_factor * lmax) + 1)
    else:
        xvals = np.asarray(xvals, dtype=np.float64)
    ls = np.arange(0, lmax + 1, dtype=np.float64)
    facs = (2 * ls + 1) / (4 * np.pi)

    if kernel == (0, 0):
        ct = facs * cl[: lmax + 1]
        corr = np.empty(len(xvals))
        for i, x in enumerate(xvals):
            P = legendre_funcs(lmax, x, (0, 0))
            corr[i] = np.dot(ct, P)
        return corr
    elif kernel not in ((0, 2), (2, 0), (2, 2), (2, -2)):
        raise ValueError(
            f"unsupported kernel {kernel!r}: only (0, 0), (0, 2)/(2, 0), "
            "(2, 2), and (2, -2) are supported"
        )

    # For polarization, all arrays start at 2
    ls2 = ls[2:]
    lfacs = ls2 * (ls2 + 1)
    lfacs2 = (ls2 + 2) * (ls2 - 1)
    lrootfacs = np.sqrt(lfacs * lfacs2)

    ct = facs[2:] * cl[2 : lmax + 1]
    corr = np.empty(len(xvals))
    for i, x in enumerate(xvals):
        d20, d22, d2m2 = legendre_funcs(lmax, x, (2, 2), lfacs, lfacs2, lrootfacs)
        d = d2m2 if kernel == (2, -2) else (d22 if kernel == (2, 2) else d20)
        corr[i] = np.dot(ct, d)
    return corr


def _corr2cl(corr, kernel, lmax=None, sampling_factor=1):
    """
    Transform a single correlation-function component (1D array) back to a
    single Cl component, via the explicit kernel selected by `kernel` --
    see `_cl2corr` for the supported kernel values and conventions. This is
    the inverse of `_cl2corr`.
    Note that using _cl2corr followed by _corr2cl is generally very accurate (< 1e-5 relative error) if
    xvals, weights = np.polynomial.legendre.leggauss(lmax+1)

    :param corr: correlation function component array, mirroring
        `_cl2corr`'s output shape
    :param kernel: (s1, s2) kernel selector; only (0, 0), (2, 0)/(0, 2),
        (2, 2), and (2, -2) are supported
    :param lmax: maximum :math:`\ell` to calculate :math:`C_\ell`
    :param sampling_factor: oversampling factor for the quadrature grid
    :return: Cl component array, the theta axis replaced by the l axis.
        Includes :math:`\ell(\ell+1)/2\pi` factors.
    """
    corr = np.asarray(corr, dtype=np.float64)

    if lmax is None:
        lmax = corr.shape[-1] - 1

    xvals, weights = _cached_gauss_legendre(int(sampling_factor * lmax) + 1)

    if kernel == (0, 0):
        cl = np.zeros(lmax + 1)
        for x, weight, c in zip(xvals, weights, corr):
            P = legendre_funcs(lmax, x, (0, 0))
            cl += (weight * c) * P
        return 2 * np.pi * cl
    elif kernel not in ((0, 2), (2, 0), (2, 2), (2, -2)):
        raise ValueError(
            f"unsupported kernel {kernel!r}: only (0, 0), (0, 2)/(2, 0), "
            "(2, 2), and (2, -2) are supported"
        )

    # For polarization, all arrays start at 2
    ls = np.arange(2, lmax + 1, dtype=np.float64)
    lfacs = ls * (ls + 1)
    lfacs2 = (ls + 2) * (ls - 1)
    lrootfacs = np.sqrt(lfacs * lfacs2)

    cl = np.zeros(lmax + 1)
    for x, weight, c in zip(xvals, weights, corr):
        d20, d22, d2m2 = legendre_funcs(lmax, x, (2, 2), lfacs, lfacs2, lrootfacs)
        d = d2m2 if kernel == (2, -2) else (d22 if kernel == (2, 2) else d20)
        cl[2:] += (weight * c) * d
    return 2 * np.pi * cl


def cl2corr(cls, progress: Progress | None = None):
    """
    Transforms cls to correlation functions
    Args:
        cls: Data Cl
        progress: optional progress reporter
    Returns:
        corr: correlation function
    """
    if progress is None:
        progress = NoProgress()

    wds = {}
    current, total = 0, len(cls)
    for key in cls.keys():
        current += 1
        progress.update(current, total)
        with progress.task(f"{key}"):
            cl = cls[key]
            spin = cl.spin
            # Grab metadata
            dtype = cl.array.dtype
            # Determine lmax from ell field or shape along ell axis
            lmax = len(get_result_array(cl, "ell")[0]) - 1
            xvals, _ = _cached_gauss_legendre(lmax + 1)
            # transform to corrs, dispatching directly on spin: add/subtract
            # into the "+/-" basis, then transform each component with its
            # own kernel
            if spin == (0, 0):
                wd = _cl2corr(cl.array, (0, 0), lmax=lmax)
            elif spin in ((0, 2), (2, 0)):
                # T x spin-2 cross correlation: both combinations use the
                # same d20 kernel
                Ta, Tb = cl.array[0], cl.array[1]
                cp, cm = Ta + Tb, Ta - Tb
                wd = np.array(
                    [
                        _cl2corr(cp, (2, 0), lmax=lmax),
                        _cl2corr(cm, (2, 0), lmax=lmax),
                    ]
                )
            else:
                # spin (2, 2): EE/BB use d22/d2m2 on the diagonal pair's
                # sum/difference, EB/BE use d22/d2m2 (negated) on the
                # anti-diagonal pair's sum/difference. Kept as one shared
                # loop (rather than 4 separate _cl2corr component calls)
                # so allP/alldP are computed once per quadrature point
                # instead of 4 times.
                ls = np.arange(0, lmax + 1, dtype=np.float64)
                facs = (2 * ls + 1) / (4 * np.pi)
                ls2 = ls[2:]
                lfacs = ls2 * (ls2 + 1)
                lfacs2 = (ls2 + 2) * (ls2 - 1)
                lrootfacs = np.sqrt(lfacs * lfacs2)
                EE, EB = cl.array[0, 0], cl.array[0, 1]
                BE, BB = cl.array[1, 0], cl.array[1, 1]
                cp = facs[2:] * (EE + BB)[2 : lmax + 1]
                cm = facs[2:] * (EE - BB)[2 : lmax + 1]
                icp = facs[2:] * (EB - BE)[2 : lmax + 1]
                icm = facs[2:] * (EB + BE)[2 : lmax + 1]
                wd = np.zeros((2, 2, len(xvals)))
                for i, x in enumerate(xvals):
                    _, d22, d2m2 = legendre_funcs(
                        lmax, x, spin, lfacs, lfacs2, lrootfacs
                    )
                    wd[0, 0, i] = np.dot(cp, d22)  # EE-like
                    wd[1, 1, i] = np.dot(cm, d2m2)  # BB-like
                    wd[0, 1, i] = -np.dot(icp, d22)  # EB-like
                    wd[1, 0, i] = -np.dot(icm, d2m2)  # BE-like
            # Add metadata back
            wd = np.array(list(wd), dtype=dtype)
            wds[key] = replace(
                cls[key],
                ell=xvals,
                array=wd,
            )
    return wds


def corr2cl(wds, progress: Progress | None = None):
    """
    Transforms correlation functions to cls
    Args:
        wds: data correlation functions
        progress: optional progress reporter
    Returns:
        corr: correlation function
    """
    if progress is None:
        progress = NoProgress()

    cls = {}
    current, total = 0, len(wds)
    for key in wds.keys():
        current += 1
        progress.update(current, total)
        with progress.task(f"{key}"):
            wd = wds[key]
            spin = wd.spin
            # Grab metadata
            dtype = wd.array.dtype
            # Derive lmax from xvals stored in the correlation's ell field
            xvals = get_result_array(wd, "ell")[0]
            weights = _cached_gauss_legendre(len(xvals))[1]
            lmax = len(xvals) - 1
            # transform to cl, dispatching directly on spin: undo each
            # component's own kernel, then add/subtract back to the
            # physical [[EE, EB], [BE, BB]] (or [Ta, Tb]) layout
            if spin == (0, 0):
                cl = _corr2cl(wd.array, (0, 0), lmax=lmax)
            elif spin in ((0, 2), (2, 0)):
                clp = _corr2cl(wd.array[0], (2, 0), lmax=lmax)
                clm = _corr2cl(wd.array[1], (2, 0), lmax=lmax)
                cl = np.array([(clp + clm) / 2, (clp - clm) / 2])
            else:
                # spin (2, 2): kept as one shared loop, mirroring cl2corr's
                # (2, 2) branch -- see there for the kernel/slot mapping.
                ls = np.arange(2, lmax + 1, dtype=np.float64)
                lfacs = ls * (ls + 1)
                lfacs2 = (ls + 2) * (ls - 1)
                lrootfacs = np.sqrt(lfacs * lfacs2)
                r = np.zeros((2, 2, lmax + 1))
                for i, (x, weight) in enumerate(zip(xvals, weights)):
                    _, d22, d2m2 = legendre_funcs(
                        lmax, x, spin, lfacs, lfacs2, lrootfacs
                    )
                    r[0, 0, 2:] += (weight * wd.array[0, 0, i]) * d22
                    r[0, 1, 2:] += (weight * wd.array[1, 1, i]) * d2m2
                    r[1, 0, 2:] += -(weight * wd.array[1, 0, i]) * d2m2
                    r[1, 1, 2:] += -(weight * wd.array[0, 1, i]) * d22
                p_diag, m_diag = r[0, 0], r[0, 1]
                p_anti, m_anti = r[1, 0], r[1, 1]
                EE, BB = (p_diag + m_diag) / 2, (p_diag - m_diag) / 2
                EB, BE = (p_anti + m_anti) / 2, (p_anti - m_anti) / 2
                cl = 2 * np.pi * np.array([[EE, EB], [BE, BB]])
            # Add metadata back
            cl = np.array(list(cl), dtype=dtype)
            cls[key] = replace(
                wds[key],
                ell=np.arange(lmax + 1),
                array=cl,
            )
    return cls
